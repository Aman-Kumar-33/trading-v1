"""The 5 minute system's candle_confirmed rule, replayed over the downloaded NIFTY 50 history (see tolerance_study.py):

  BUY   at the close of the first 5m candle (from 09:20, before 15:00) whose MACD is more than the tolerance above the signal line
        after the lines were last at/below each other; tolerance = max(0.02, 0.02 x the stock's ATR)
  SELL  at the close of the first candle with MACD below the signal line (no tolerance), at the 15:00 square-off, or when the
        price touches a stop `stop_pct` below the entry (checked on the 1-minute lows)

    python scripts/tolerance_study_candle_rule.py

Costs: 7 bp per round trip, the same as the other studies. Prints trades per stock-day, gross and net, with day-cluster bootstrap CIs.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.core.config import load_settings  # noqa: E402
from backend.data import trending  # noqa: E402
from backend.research import live_macd_replay as R  # noqa: E402
from tolerance_study import stock_data  # noqa: E402

COST = 7e-4
SQUARE_OFF_MIN = 15 * 60
FIRST_TRADE_MIN = 9 * 60 + 20


def simulate(df1: pd.DataFrame, bs: R.BarState, k_atr: float, floor: float, stop_pct: float | None):
    """Trades as (day, entry, exit, reason)."""
    bars = bs.bars
    day, end_min = bars["day"].to_numpy(), bars["end_min"].to_numpy()
    close = bars["close"].to_numpy(float)
    gap, atr = bs.gap, bs.atr
    tl = (pd.to_datetime(df1["ts"], utc=True).dt.tz_convert("Asia/Kolkata"))
    t1 = tl.dt.tz_localize(None).to_numpy("datetime64[m]")
    low, op = df1["low"].to_numpy(float), df1["open"].to_numpy(float)
    day_start = pd.to_datetime(bars["day"]).to_numpy("datetime64[D]")
    bar_end_t = day_start + end_min.astype("timedelta64[m]")

    out, armed, pos = [], False, None          # pos = (entry px, stop px, entry bar index)
    for i in range(len(bars)):
        g = gap[i]
        if np.isnan(g):
            continue
        if pos is not None:
            # the stop, on the 1-minute bars since the entry candle closed
            if stop_pct:
                a = np.searchsorted(t1, bar_end_t[pos[2]], "left")
                b = np.searchsorted(t1, bar_end_t[i], "left")
                hit = np.flatnonzero(low[a:b] <= pos[1])
                if len(hit):
                    px = min(op[a + hit[0]], pos[1])
                    out.append((day[i], pos[0], px, "stop"))
                    pos = None
            if pos is not None and (end_min[i] >= SQUARE_OFF_MIN or g < 0 or day[i] != day[pos[2]]):
                out.append((day[pos[2]], pos[0], close[i] if day[i] == day[pos[2]] else close[i - 1], "signal" if g < 0 else "square_off"))
                pos = None
        tol = max(floor, k_atr * atr[i]) if not np.isnan(atr[i]) and atr[i] > 0 else floor
        if g <= 0:
            armed = True
        elif g > tol:
            if armed and pos is None and FIRST_TRADE_MIN <= end_min[i] < SQUARE_OFF_MIN:
                pos = (close[i], close[i] * (1 - stop_pct) if stop_pct else -1.0, i)
            armed = False
    return out


def boot(x: np.ndarray, B=4000):
    r = np.random.default_rng(5).integers(0, len(x), (B, len(x)))
    m = x[r].mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def main() -> int:
    settings = load_settings()
    root = settings.paths.root / "research_data" / "angel_1m"
    symbols = trending.universe_symbols(settings)
    configs = {"plain cross, no tolerance, no stop": (0.0, 0.0, None),
               "tolerance 0.02 + 0.02xATR, plain exit, no stop": (0.02, 0.02, None),
               "tolerance 0.02 + 0.02xATR, plain exit, 5% stop  <- the 5 minute system now": (0.02, 0.02, 0.05)}
    rows = {c: [] for c in configs}
    n_days = {}
    for n, sym in enumerate(symbols, 1):
        got = stock_data(root, sym)
        if got is None:
            continue
        df1, ok = got
        bs = R.bar_state(R.to_bars(df1, 5))
        n_days[sym] = len(ok)
        for c, (floor, k, stop) in configs.items():
            for d, e, x, why in simulate(df1, bs, k, floor, stop):
                if d in ok:
                    rows[c].append((sym, d, x / e - 1, why))
        print(f"  [{n}/{len(symbols)}] {sym}", flush=True)
    all_days = sorted({d for sym in symbols for d in []} | set())
    n_stockdays = sum(n_days.values())
    print(f"\n{len(n_days)} stocks, {n_stockdays} stock-days (7 bp round-trip charges)\n")
    for c, recs in rows.items():
        df = pd.DataFrame(recs, columns=["symbol", "date", "ret", "why"])
        per_day = (df.assign(net=df["ret"] - COST).groupby("date")["net"].sum()
                   .reindex(sorted({d for r in rows.values() for _, d, _, _ in r}), fill_value=0.0))
        g = df["ret"] * 1e4
        m, lo, hi = boot(per_day.to_numpy() / len(n_days) * 1e4)
        print(f"{c}\n    trades per stock-day {len(df) / n_stockdays:.2f} | gross {g.mean():+.2f} bp/trade | net {1e4 * (df['ret'] - COST).mean():+.2f} bp/trade"
              f" | net {m:+.2f} bp per stock-day [{lo:+.2f}, {hi:+.2f}] | win {100 * (df['ret'] > COST).mean():.0f}% | exits: "
              + ", ".join(f"{k} {v}" for k, v in df["why"].value_counts().items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
