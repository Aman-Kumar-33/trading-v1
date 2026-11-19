"""Replay the live (tick-driven) MACD rule of `SimpleRunner._live_tick` over history, for many tolerance / hold settings.

Angel One has no historical ticks, so a minute bar is turned into four pseudo-ticks (open, the extreme reached first,
the other extreme, close). The rule is exactly the live one:
    gap(price)  = MACD - signal of the FORMING bar, if it closed at `price`  (EMA state of the last closed bar)
    tolerance   = max(floor, k x ATR of the last closed bar)
    BUY  when gap > tolerance  and the state is "down" and nothing is held
    SELL when gap < -tolerance and (a long is held or the state is "up")
    an order is placed only once that condition has held for `hold` seconds without interruption.
State at the start of a day = side of the last closed bar whose |gap| > floor (what a fresh process computes).
Everything is intraday: no entry at/after 15:00, anything still open is sold at the last price before 15:00.

`simulate` walks RUNS of ticks that are beyond the band on the same side instead of every tick, which is exact (the rule
only changes anything at a run's start or when its hold time is reached) and fast enough to scan a large grid.
`reference_simulate` is the plain per-tick loop used to prove that in tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from backend.features import indicators as ind

A12, A26, A9 = 2 / 13, 2 / 27, 2 / 10
SESSION_OPEN_MIN = 9 * 60 + 15
SQUARE_OFF_MIN = 15 * 60
FEED_END_MIN = 15 * 60 + 15                # the last minute Angel's 1-minute history really has is 15:14
TICK_OFFSETS = (0.0, 20.0, 40.0, 59.0)   # seconds into the minute at which the four pseudo-ticks are placed


# ---------------------------------------------------------------------- data
def load_1m(stock_dir: Path, fill_close: bool = True) -> pd.DataFrame:
    """All downloaded 1-minute windows of one stock (research_data/angel_1m/<SYMBOL>/*.parquet), one clean series."""
    parts = [pd.read_parquet(p) for p in sorted(stock_dir.glob("*.parquet"))]
    if not parts:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "synthetic"])
    df = pd.concat(parts, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert("Asia/Kolkata")
    df = df.sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
    mod = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    # Angel's 1-minute feed has no bars from 15:15 to 15:28 and a single odd 15:29 row: keep what is real (09:15-15:14)
    df = df[(mod >= SESSION_OPEN_MIN) & (mod < FEED_END_MIN)].reset_index(drop=True)
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    df["synthetic"] = False
    if fill_close and not df.empty:
        # ...and fill 15:15-15:29 with flat bars at the last close, so the 5m/15m bars of the closing minutes exist and the
        # EMAs carry to the next morning on a normal series. Nothing is traded after 15:00, only the EMA state is affected.
        last = df.groupby(df["ts"].dt.date).tail(1)
        extra = []
        for _, r in last.iterrows():
            for m in range(FEED_END_MIN, 15 * 60 + 30):
                ts = r["ts"].normalize() + pd.Timedelta(minutes=m)
                extra.append({"ts": ts, "open": r["close"], "high": r["close"], "low": r["close"], "close": r["close"], "synthetic": True})
        df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True).sort_values("ts").reset_index(drop=True)
    return df[["ts", "open", "high", "low", "close", "synthetic"]]


def to_bars(df1: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Bars of `minutes` aligned to the 09:15 open, with the day and the minute the bar ends at."""
    mod = (df1["ts"].dt.hour * 60 + df1["ts"].dt.minute).to_numpy()
    day = df1["ts"].dt.date
    bucket = (mod - SESSION_OPEN_MIN) // minutes
    g = df1.groupby([day.rename("day"), pd.Series(bucket, index=df1.index, name="b")], sort=True)   # same index: a slice of a frame works too
    bars = g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), ts=("ts", "first")).reset_index()
    bars["start_min"] = SESSION_OPEN_MIN + bars["b"] * minutes
    bars["end_min"] = bars["start_min"] + minutes
    return bars.sort_values(["day", "b"]).reset_index(drop=True)


@dataclass
class BarState:
    bars: pd.DataFrame
    e12: np.ndarray
    e26: np.ndarray
    sig: np.ndarray
    atr: np.ndarray
    gap: np.ndarray        # closed-bar MACD - signal


def bar_state(bars: pd.DataFrame) -> BarState:
    close = bars["close"].astype(float)
    m = ind.macd(close)
    return BarState(bars, ind.ema(close, 12).to_numpy(), ind.ema(close, 26).to_numpy(), m["signal"].to_numpy(),
                    ind.atr(bars["high"], bars["low"], close, 14).to_numpy(), (m["macd"] - m["signal"]).to_numpy())


@dataclass
class Ticks:
    gap: np.ndarray        # forming-bar MACD - signal at each tick
    atr: np.ndarray        # ATR of the last closed bar (0 where unknown)
    secs: np.ndarray       # seconds (epoch)
    px: np.ndarray
    day: np.ndarray        # day index per tick (int)
    state0: dict           # day index -> "up" | "down" | None, as a fresh process would start it
    day_labels: list
    bar: np.ndarray | None = None   # index (into BarState) of the last CLOSED bar at each tick


def pseudo_ticks(df1: pd.DataFrame, bs: BarState, minutes: int, floor: float, order: str = "auto") -> Ticks:
    """Four pseudo-ticks per minute bar, each with the forming-bar MACD gap computed from the last CLOSED `minutes` bar."""
    bars = bs.bars
    mod = (df1["ts"].dt.hour * 60 + df1["ts"].dt.minute).to_numpy()
    keep = mod < SQUARE_OFF_MIN                      # nothing happens at/after 15:00
    d1 = df1[keep].reset_index(drop=True)
    mod = mod[keep]
    day_key = d1["ts"].dt.date
    bucket = (mod - SESSION_OPEN_MIN) // minutes
    lookup = {(d, b): i for i, (d, b) in enumerate(zip(bars["day"], bars["b"]))}
    j = np.fromiter((lookup[(d, b)] for d, b in zip(day_key, bucket)), dtype=np.int64, count=len(d1))
    o, h, l, c = (d1[x].to_numpy() for x in ("open", "high", "low", "close"))
    if order == "auto":
        first_low = c >= o                           # a rising minute is assumed to dip first, a falling one to spike first
    elif order == "low_first":
        first_low = np.ones(len(d1), bool)
    else:
        first_low = np.zeros(len(d1), bool)
    x1, x2 = np.where(first_low, l, h), np.where(first_low, h, l)
    px = np.stack([o, x1, x2, c], axis=1).reshape(-1)
    minute_start = ((d1["ts"].dt.tz_convert("UTC").dt.tz_localize(None) - pd.Timestamp("1970-01-01")) /
                    pd.Timedelta(seconds=1)).to_numpy(dtype=float)
    secs = (minute_start[:, None] + np.array(TICK_OFFSETS)[None, :]).reshape(-1)
    jj = np.repeat(j, 4)
    prev = jj - 1
    ok = prev >= 0
    pv = np.where(ok, prev, 0)
    e12b, e26b, sgb, atrb = bs.e12[pv], bs.e26[pv], bs.sig[pv], np.nan_to_num(bs.atr[pv], nan=0.0)
    m = (A12 * px + (1 - A12) * e12b) - (A26 * px + (1 - A26) * e26b)
    gap = (1 - A9) * (m - sgb)
    # a forming bar is only usable when the bar before it is the previous bar of the same day, or the last bar of the day before
    gap = np.where(ok & np.isfinite(gap), gap, np.nan)
    days = bars["day"].to_numpy()
    uniq = list(dict.fromkeys(days))
    day_idx = {d: i for i, d in enumerate(uniq)}
    day_per_bar = np.array([day_idx[d] for d in days])
    day_t = np.repeat(day_per_bar[j], 4)
    # state at the start of each day: side of the last closed bar before it with |gap| > floor
    sign = np.where(np.abs(bs.gap) > floor, np.sign(bs.gap), 0.0)
    sign = np.where(np.isnan(bs.gap), 0.0, sign)
    last = pd.Series(np.where(sign != 0, sign, np.nan)).ffill().to_numpy()
    first_bar_of_day = {}
    for i, d in enumerate(day_per_bar):
        first_bar_of_day.setdefault(d, i)
    state0 = {}
    for d, i in first_bar_of_day.items():
        v = last[i - 1] if i > 0 else np.nan
        state0[d] = None if np.isnan(v) else ("up" if v > 0 else "down")
    good = np.isfinite(gap)
    return Ticks(gap[good], atrb[good], secs[good], px[good], day_t[good], state0, uniq, pv[good])


# ---------------------------------------------------------------------- the rule
Trade = tuple   # (day index, entry px, exit px, entry secs, exit secs, entry tick index, exit tick index)


@dataclass
class Rule:
    """Optional extras on top of the plain tolerance + hold rule - each is a way of ignoring whipsaw trades."""
    exit_k: float | None = None        # tolerance of a SELL as a multiple of ATR (None = the same as the BUY's)
    exit_hold: float | None = None     # seconds a SELL must hold (None = the same as `hold`)
    gate: np.ndarray | None = None     # per tick; False = a BUY is not allowed now (that cross is ignored)
    cooldown: float = 0.0              # seconds after an exit before the next BUY is allowed
    max_entries: int | None = None     # BUYs allowed per day


def simulate(tk: Ticks, k: float, hold: float, floor: float = 0.02, rule: Rule | None = None) -> list[Trade]:
    """All trades of the rule over `tk` (see module doc). Exact and run-based."""
    n = len(tk.px)
    if n == 0:
        return []
    rule = rule or Rule()
    tol_in = np.maximum(floor, k * tk.atr)
    tol_out = tol_in if rule.exit_k is None else np.maximum(floor, rule.exit_k * tk.atr)
    hold_out = hold if rule.exit_hold is None else rule.exit_hold
    s = np.where(tk.gap > tol_in, 1, np.where(tk.gap < -tol_out, -1, 0)).astype(np.int8)
    brk = np.r_[True, (s[1:] != s[:-1]) | (tk.day[1:] != tk.day[:-1])]
    starts = np.flatnonzero(brk)
    ends = np.r_[starts[1:] - 1, n - 1]
    trades: list[Trade] = []
    cur_day, state, pos, pos_secs, pos_i = None, None, None, 0.0, 0
    cool_until, entries = -np.inf, 0
    secs, px, day = tk.secs, tk.px, tk.day
    last_of_day = {}
    dstarts = np.flatnonzero(np.r_[True, day[1:] != day[:-1]])
    dends = np.r_[dstarts[1:] - 1, n - 1]
    for a, b in zip(dstarts, dends):
        last_of_day[int(day[a])] = int(b)

    def close_day(d):
        nonlocal pos
        if pos is not None:
            i = last_of_day[d]
            trades.append((d, pos, px[i], pos_secs, secs[i], pos_i, i))
            pos = None

    for a, b in zip(starts, ends):
        d = int(day[a])
        if d != cur_day:
            if cur_day is not None:
                close_day(cur_day)
            cur_day, state, cool_until, entries = d, tk.state0.get(d), -np.inf, 0
        side = s[a]
        if side == 0:
            continue
        h = hold if side == 1 else hold_out
        # index of the tick that fires, or -1 if the run ends before the hold time is reached
        if h <= 0:
            f = a
        else:
            f = a + int(np.searchsorted(secs[a:b + 1], secs[a] + h, side="left"))
            if f > b:
                f = -1
        if side == 1:
            if pos is None and state == "down":
                if f >= 0:
                    if ((rule.gate is None or rule.gate[f]) and secs[f] >= cool_until
                            and (rule.max_entries is None or entries < rule.max_entries)):
                        pos, pos_secs, pos_i, state = px[f], secs[f], f, "up"
                        entries += 1
                    else:
                        state = "up"      # this cross is ignored
            else:
                state = "up"
        else:
            if pos is not None or state == "up":
                if f >= 0:
                    if pos is not None:
                        trades.append((d, pos, px[f], pos_secs, secs[f], pos_i, f))
                        pos = None
                        cool_until = secs[f] + rule.cooldown
                    state = "down"
            else:
                state = "down"
    if cur_day is not None:
        close_day(cur_day)
    return trades


def reference_simulate(tk: Ticks, k: float, hold: float, floor: float = 0.02, rule: Rule | None = None) -> list[Trade]:
    """The same rule, one tick at a time, written like `SimpleRunner._live_tick` - the check on `simulate`."""
    rule = rule or Rule()
    hold_out = hold if rule.exit_hold is None else rule.exit_hold
    trades: list[Trade] = []
    n = len(tk.px)
    cur_day, state, pos, pos_secs, pos_i, pend = None, None, None, 0.0, 0, None
    cool_until, entries = -np.inf, 0
    for i in range(n):
        d = int(tk.day[i])
        if d != cur_day:
            if pos is not None:
                j = i - 1
                trades.append((cur_day, pos, tk.px[j], pos_secs, tk.secs[j], pos_i, j))
                pos = None
            cur_day, state, pend, cool_until, entries = d, tk.state0.get(d), None, -np.inf, 0
        gap = tk.gap[i]
        tol_in = max(floor, k * tk.atr[i])
        tol_out = tol_in if rule.exit_k is None else max(floor, rule.exit_k * tk.atr[i])
        want = None
        if gap > tol_in:
            if state == "down" and pos is None:
                want = 1
            else:
                state = "up"
        elif gap < -tol_out:
            if pos is not None or state == "up":
                want = -1
            else:
                state = "down"
        if want is None:
            pend = None
            continue
        if pend is None or pend[0] != want:
            pend = (want, tk.secs[i])
        if tk.secs[i] - pend[1] < (hold if want == 1 else hold_out):
            continue
        pend = None
        if want == 1:
            if ((rule.gate is None or rule.gate[i]) and tk.secs[i] >= cool_until
                    and (rule.max_entries is None or entries < rule.max_entries)):
                pos, pos_secs, pos_i, state = tk.px[i], tk.secs[i], i, "up"
                entries += 1
            else:
                state = "up"
        else:
            if pos is not None:
                trades.append((d, pos, tk.px[i], pos_secs, tk.secs[i], pos_i, i))
                pos = None
                cool_until = tk.secs[i] + rule.cooldown
            state = "down"
    if pos is not None:
        trades.append((cur_day, pos, tk.px[n - 1], pos_secs, tk.secs[n - 1], pos_i, n - 1))
    return trades


# ---------------------------------------------------------------------- "the lines are too close for too long": bar features
def chop_count(gap: np.ndarray, window: int) -> np.ndarray:
    """Per closed bar: how many times the MACD/signal gap changed sign over the last `window` bars."""
    sign = np.sign(np.where(np.isnan(gap), 0.0, gap))
    flips = np.r_[0.0, (sign[1:] * sign[:-1] < 0).astype(float)]
    return pd.Series(flips).rolling(window, min_periods=1).sum().to_numpy()


def hug_ratio(gap: np.ndarray, atr: np.ndarray, window: int) -> np.ndarray:
    """Per closed bar: mean |gap| / ATR over the last `window` bars - small = the lines have been hugging each other."""
    r = np.abs(gap) / np.where(atr > 0, atr, np.nan)
    return pd.Series(r).rolling(window, min_periods=max(window // 2, 1)).mean().to_numpy()


# ---------------------------------------------------------------------- baselines: act on closed bars only
def candle_close_trades(bs: BarState, df1: pd.DataFrame, minutes: int) -> list[Trade]:
    """The bar-close rule: BUY on the bar whose close is the first with MACD above signal (a cross up), SELL on a cross
    down; no BUY on a bar that ends at/after 15:00; whatever is open at 15:00 is sold. Returns (day index, entry, exit, ...)."""
    bars = bs.bars
    days = bars["day"].to_numpy()
    close, end_min, gap = bars["close"].to_numpy(), bars["end_min"].to_numpy(), bs.gap
    mod = (df1["ts"].dt.hour * 60 + df1["ts"].dt.minute).to_numpy()
    lastpx = df1[mod < SQUARE_OFF_MIN].groupby(df1["ts"].dt.date)["close"].last().to_dict()
    trades: list[Trade] = []
    pos, cur = None, None
    for i in range(1, len(bars)):
        d = days[i]
        if d != cur:
            if pos is not None:
                trades.append((cur, pos, lastpx[cur], 0.0, 0.0))
                pos = None
            cur = d
        if np.isnan(gap[i]) or np.isnan(gap[i - 1]):
            continue
        t = float(end_min[i])
        if t > SQUARE_OFF_MIN:
            continue
        cross_up, cross_dn = gap[i - 1] <= 0 < gap[i], gap[i - 1] >= 0 > gap[i]
        if pos is None and cross_up and t < SQUARE_OFF_MIN:
            pos = close[i]
        elif pos is not None and cross_dn:
            trades.append((d, pos, close[i], 0.0, 0.0))
            pos = None
    if pos is not None:
        trades.append((cur, pos, lastpx[cur], 0.0, 0.0))
    return trades
