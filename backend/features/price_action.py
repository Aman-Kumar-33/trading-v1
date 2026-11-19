"""Price-action structure: swing points, market structure (BOS/CHoCH), order blocks, fair value gaps,
liquidity sweeps, inside/outside bars, pin bars, and support/resistance zones.

Lookahead safety: a swing point at bar i is only *confirmed* once `right` bars later have printed (you
need to see the bars after a pivot to know it was a pivot). Every function here that reports "as of this
bar" state (trend, BOS/CHoCH, order blocks, nearest S/R) only uses swings that were already confirmed by
that bar - never the bar's own not-yet-confirmed pivot. This is what makes it safe to use identically in
a live strategy and in a backtest.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------- swing points
def swings(high: pd.Series, low: pd.Series, left: int = 2, right: int = 2) -> pd.DataFrame:
    """Fractal swing points: bar i is a swing high if high[i] is (tying allowed) the max over
    [i-left, i+right], and a swing low if low[i] is the min over the same window. On a tie (a flat
    top/bottom spanning several bars - rare with real market data, which is essentially continuous),
    every bar sharing the extreme is marked.

    Marked at the pivot bar itself (not lookahead-safe on its own - see `confirmed_at`, which shifts
    the mark to the bar where it becomes knowable).
    """
    h, l = high.to_numpy(dtype=float), low.to_numpy(dtype=float)
    n = len(h)
    sh = np.zeros(n, dtype=bool)
    sl = np.zeros(n, dtype=bool)
    for i in range(left, n - right):
        seg_h = h[i - left: i + right + 1]
        seg_l = l[i - left: i + right + 1]
        if h[i] == seg_h.max():
            sh[i] = True
        if l[i] == seg_l.min():
            sl[i] = True
    out = pd.DataFrame({"swing_high": sh, "swing_low": sl}, index=high.index)
    out["swing_high_price"] = high.where(out["swing_high"])
    out["swing_low_price"] = low.where(out["swing_low"])
    return out


def confirmed_at(mark: pd.Series, right: int) -> pd.Series:
    """Shift a pivot-bar boolean mark forward by `right` bars - the earliest point a strategy or
    backtest may legally act on it."""
    return mark.shift(right).fillna(False).astype(bool)


# ---------------------------------------------------------------------- inside / outside bars
def inside_outside_bars(high: pd.Series, low: pd.Series) -> pd.DataFrame:
    prev_high, prev_low = high.shift(1), low.shift(1)
    inside = (high <= prev_high) & (low >= prev_low)
    outside = (high >= prev_high) & (low <= prev_low) & ~((high == prev_high) & (low == prev_low))
    return pd.DataFrame({"inside_bar": inside.fillna(False), "outside_bar": outside.fillna(False)})


# ---------------------------------------------------------------------- pin bars
def pin_bar(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
           min_wick_body_ratio: float = 2.0, max_opposite_wick_frac: float = 0.3) -> pd.Series:
    """+1 = bullish pin bar (long lower wick, small body/upper wick - rejection of lower prices),
    -1 = bearish pin bar (long upper wick), 0 = neither. Shape only, no trend/location context -
    combine with support_resistance zones (below) to get "pin bar at S/R"."""
    body = (close - open_).abs()
    rng = (high - low).replace(0, np.nan)
    body_safe = body.mask(body == 0, rng * 0.001)     # avoid divide-by-zero for doji-like candles
    upper = high - pd.concat([open_, close], axis=1).max(axis=1)
    lower = pd.concat([open_, close], axis=1).min(axis=1) - low
    bullish = (lower >= min_wick_body_ratio * body_safe) & (upper <= max_opposite_wick_frac * rng)
    bearish = (upper >= min_wick_body_ratio * body_safe) & (lower <= max_opposite_wick_frac * rng)
    return pd.Series(np.select([bullish.fillna(False), bearish.fillna(False)], [1, -1], default=0), index=open_.index)


# ---------------------------------------------------------------------- fair value gaps
def fair_value_gaps(high: pd.Series, low: pd.Series) -> pd.DataFrame:
    """3-candle imbalance. Marked at the 3rd (confirming) candle - the gap is only knowable then.
    bullish: low[i+1] > high[i-1]   (candle i-1's high never reached by candle i+1's low)
    bearish: high[i+1] < low[i-1]
    """
    prev_high, prev_low = high.shift(2), low.shift(2)
    bullish = low > prev_high
    bearish = high < prev_low
    top = pd.Series(np.where(bullish, low, np.where(bearish, prev_low, np.nan)), index=high.index)
    bottom = pd.Series(np.where(bullish, prev_high, np.where(bearish, high, np.nan)), index=high.index)
    return pd.DataFrame({"bullish_fvg": bullish.fillna(False), "bearish_fvg": bearish.fillna(False),
                        "fvg_top": top, "fvg_bottom": bottom})


# ---------------------------------------------------------------------- support / resistance
@dataclass
class Zone:
    level: float
    touches: int
    kind: str        # "support" | "resistance" | "both"
    last_touch: int  # positional index of the most recent touch


def cluster_zones(prices: pd.Series, atr_at_prices: pd.Series | float, cluster_atr_mult: float = 0.5,
                  min_touches: int = 2, max_zones: int = 15) -> list[Zone]:
    """Cluster a set of (already-confirmed) swing prices into support/resistance zones.

    Greedy clustering: sort by price, merge points within `cluster_atr_mult` * ATR of the running
    cluster mean. Zones with fewer than `min_touches` are dropped. `prices` should be an on-demand
    snapshot (e.g. confirmed swing prices up to "now") - this is not a per-bar rolling function; the
    caller controls lookahead safety by only passing prices confirmed as of the bar being evaluated.
    """
    if prices.empty:
        return []
    atr_val = float(atr_at_prices) if np.isscalar(atr_at_prices) else float(pd.Series(atr_at_prices).iloc[-1])
    tol = max(cluster_atr_mult * atr_val, prices.abs().median() * 1e-6)
    order = prices.sort_values()
    clusters: list[list[float]] = []
    touch_idx: list[list[int]] = []
    for idx, price in order.items():
        if clusters and abs(price - np.mean(clusters[-1])) <= tol:
            clusters[-1].append(price)
            touch_idx[-1].append(idx)
        else:
            clusters.append([price])
            touch_idx.append([idx])
    zones = []
    for c, ti in zip(clusters, touch_idx):
        if len(c) >= min_touches:
            zones.append(Zone(level=float(np.mean(c)), touches=len(c), kind="unclassified", last_touch=max(ti)))
    zones.sort(key=lambda z: z.touches, reverse=True)
    return zones[:max_zones]


def rolling_nearest_levels(close: pd.Series, swing_high_confirmed: pd.Series, swing_low_confirmed: pd.Series,
                           swing_high_price: pd.Series, swing_low_price: pd.Series, lookback: int = 10
                           ) -> pd.DataFrame:
    """Per-bar nearest support (below close) and resistance (above close) from the last `lookback`
    confirmed swing points on each side. Lookahead-safe: only swings already confirmed at bar i are
    considered for bar i."""
    n = len(close)
    c = close.to_numpy(dtype=float)
    hi_conf = swing_high_confirmed.to_numpy(dtype=bool)
    lo_conf = swing_low_confirmed.to_numpy(dtype=bool)
    hi_price = swing_high_price.to_numpy(dtype=float)
    lo_price = swing_low_price.to_numpy(dtype=float)
    resistance = np.full(n, np.nan)
    support = np.full(n, np.nan)
    highs_seen: list[float] = []
    lows_seen: list[float] = []
    for i in range(n):
        if hi_conf[i]:
            highs_seen.append(hi_price[i])
            if len(highs_seen) > lookback:
                highs_seen.pop(0)
        if lo_conf[i]:
            lows_seen.append(lo_price[i])
            if len(lows_seen) > lookback:
                lows_seen.pop(0)
        above = [p for p in highs_seen if p >= c[i]]
        below = [p for p in lows_seen if p <= c[i]]
        if above:
            resistance[i] = min(above)
        if below:
            support[i] = max(below)
    return pd.DataFrame({"nearest_support": support, "nearest_resistance": resistance}, index=close.index)


# ---------------------------------------------------------------------- market structure (BOS / CHoCH / OB / sweeps)
def analyze_structure(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
                      left: int = 2, right: int = 2, impulse_atr_mult: float = 1.5,
                      atr: pd.Series | None = None) -> pd.DataFrame:
    """Single stateful pass building trend bias, BOS/CHoCH events, the nearest unmitigated order block on
    each side, and liquidity sweeps. One pass so everything shares the same swing sequence.

    trend        : running bias, +1 (bullish structure: higher highs & higher lows), -1 (bearish), 0 (none yet)
    bos          : this bar's close broke structure *with* the existing trend (continuation)
    choch        : this bar's close broke structure *against* the existing trend (possible reversal)
    bull_ob_low/high : nearest unmitigated bullish order block zone (None if none active)
    bear_ob_low/high : nearest unmitigated bearish order block zone
    sweep_high/low   : liquidity sweep of a recent swing high/low (wick through, close back inside)
    """
    n = len(close)
    sw = swings(high, low, left, right)
    hi_conf = confirmed_at(sw["swing_high"], right).to_numpy()
    lo_conf = confirmed_at(sw["swing_low"], right).to_numpy()
    hi_price_conf = sw["swing_high_price"].shift(right).to_numpy()
    lo_price_conf = sw["swing_low_price"].shift(right).to_numpy()

    if atr is None:
        from backend.features.indicators import atr as _atr
        atr = _atr(high, low, close, 14)
    atr_v = atr.to_numpy(dtype=float)
    o, h, l, c = open_.to_numpy(dtype=float), high.to_numpy(dtype=float), low.to_numpy(dtype=float), close.to_numpy(dtype=float)

    trend = np.zeros(n, dtype=int)
    bos = np.zeros(n, dtype=bool)
    choch = np.zeros(n, dtype=bool)
    sweep_high = np.zeros(n, dtype=bool)
    sweep_low = np.zeros(n, dtype=bool)
    bull_ob_lo = np.full(n, np.nan)
    bull_ob_hi = np.full(n, np.nan)
    bear_ob_lo = np.full(n, np.nan)
    bear_ob_hi = np.full(n, np.nan)

    last_swing_high = last_swing_low = np.nan
    cur_trend = 0
    active_bull_ob: tuple[float, float] | None = None   # (low, high) of candle body
    active_bear_ob: tuple[float, float] | None = None
    recent_highs: list[float] = []   # for sweep detection - unbroken recent swing highs
    recent_lows: list[float] = []

    for i in range(n):
        # 1. absorb newly confirmed swings
        if hi_conf[i] and not np.isnan(hi_price_conf[i]):
            last_swing_high = hi_price_conf[i]
            recent_highs.append(hi_price_conf[i])
            recent_highs = recent_highs[-5:]
        if lo_conf[i] and not np.isnan(lo_price_conf[i]):
            last_swing_low = lo_price_conf[i]
            recent_lows.append(lo_price_conf[i])
            recent_lows = recent_lows[-5:]

        # 2. structure breaks on this bar's close
        if not np.isnan(last_swing_high) and c[i] > last_swing_high:
            if cur_trend <= 0 and cur_trend != 0:
                choch[i] = True
            else:
                bos[i] = True
            cur_trend = 1
            last_swing_high = np.nan  # broken level is consumed; wait for the next one
        elif not np.isnan(last_swing_low) and c[i] < last_swing_low:
            if cur_trend >= 0 and cur_trend != 0:
                choch[i] = True
            else:
                bos[i] = True
            cur_trend = -1
            last_swing_low = np.nan
        trend[i] = cur_trend

        # 3. order blocks: last opposite-colour candle before a strong displacement in the new direction.
        # Mitigation (price trading back through a zone uses it up) is checked against the OB as it
        # stood *before* this bar, so a freshly-created OB is never wiped out by its own creating bar's
        # wick (a displacement candle moving away from the zone naturally wicks past it on its own side).
        if active_bull_ob is not None and l[i] < active_bull_ob[0]:
            active_bull_ob = None
        if active_bear_ob is not None and h[i] > active_bear_ob[1]:
            active_bear_ob = None
        if i >= 1 and not np.isnan(atr_v[i]):
            displacement_up = (c[i] - o[i]) > impulse_atr_mult * atr_v[i]
            displacement_down = (o[i] - c[i]) > impulse_atr_mult * atr_v[i]
            if displacement_up and c[i - 1] < o[i - 1]:
                active_bull_ob = (min(o[i - 1], c[i - 1]), max(o[i - 1], c[i - 1]))
            if displacement_down and c[i - 1] > o[i - 1]:
                active_bear_ob = (min(o[i - 1], c[i - 1]), max(o[i - 1], c[i - 1]))
        if active_bull_ob is not None:
            bull_ob_lo[i], bull_ob_hi[i] = active_bull_ob
        if active_bear_ob is not None:
            bear_ob_lo[i], bear_ob_hi[i] = active_bear_ob

        # 4. liquidity sweeps: wick through a recent (already-confirmed, still-unbroken) swing, close back inside
        for lvl in recent_highs:
            if h[i] > lvl > c[i]:
                sweep_high[i] = True
                break
        for lvl in recent_lows:
            if l[i] < lvl < c[i]:
                sweep_low[i] = True
                break

    return pd.DataFrame({
        "trend": trend, "bos": bos, "choch": choch,
        "bull_ob_low": bull_ob_lo, "bull_ob_high": bull_ob_hi,
        "bear_ob_low": bear_ob_lo, "bear_ob_high": bear_ob_hi,
        "sweep_high": sweep_high, "sweep_low": sweep_low,
    }, index=close.index)
