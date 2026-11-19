"""Candlestick pattern detectors - vectorized, ATR-normalized thresholds (tunable per stock/volatility
regime rather than TA-Lib's fixed thresholds).

Every detector returns a signed Series aligned to the input index: +100 = bullish signal on that bar,
-100 = bearish, 0 = none - the same convention TA-Lib's CDL* functions use, which is what lets
`tests/test_candlestick.py` cross-check our detectors against TA-Lib on real historical data (our
detectors won't match TA-Lib bar-for-bar since ours are ATR-tunable and TA-Lib's are fixed-threshold,
but they must broadly agree on the clear-cut cases).

Multi-candle patterns are marked on the *last* candle of the pattern (the confirming candle) - no
lookahead: everything here only reads bar i and earlier.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backend.features.indicators import atr as _atr
from backend.features.indicators import ema

BULLISH, BEARISH, NONE = 100, -100, 0


@dataclass
class Geometry:
    """Precomputed per-bar shape measurements, shared by every pattern detector so they aren't each
    recomputing body/wick/ATR from scratch."""
    open: pd.Series
    high: pd.Series
    low: pd.Series
    close: pd.Series
    body: pd.Series          # |close - open|
    top: pd.Series           # max(open, close)
    bot: pd.Series           # min(open, close)
    upper: pd.Series         # high - top
    lower: pd.Series         # bot - low
    range_: pd.Series        # high - low
    is_bull: pd.Series
    is_bear: pd.Series
    atr: pd.Series
    trend: pd.Series         # short-term prior trend: +1 up, -1 down, 0 flat/undefined (see prior_trend)


def prior_trend(close: pd.Series, lookback: int = 5) -> pd.Series:
    """+1 if the short EMA has risen over the last `lookback` bars, -1 if it fell, 0 if flat/undefined.
    A lightweight proxy for "was this candle preceded by an uptrend/downtrend" - used to disambiguate
    shape-identical reversal patterns (Hammer vs Hanging Man, Inverted Hammer vs Shooting Star, ...).
    Uses only bar i and earlier - safe for live use and backtesting alike.
    """
    fast = ema(close, max(2, lookback))
    delta = fast - fast.shift(lookback)
    return pd.Series(np.sign(delta.fillna(0)), index=close.index).astype(int)


def compute_geometry(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
                     atr: pd.Series | None = None, trend_lookback: int = 5) -> Geometry:
    body = (close - open_).abs()
    top = pd.concat([open_, close], axis=1).max(axis=1)
    bot = pd.concat([open_, close], axis=1).min(axis=1)
    upper = high - top
    lower = bot - low
    rng = high - low
    atr_s = atr if atr is not None else _atr(high, low, close, 14)
    return Geometry(open_, high, low, close, body, top, bot, upper, lower, rng,
                    (close > open_), (close < open_), atr_s, prior_trend(close, trend_lookback))


def _safe(x: pd.Series, floor_frac_of: pd.Series, frac: float = 0.05) -> pd.Series:
    """Avoid divide-by-zero on doji-like candles: floor a near-zero denominator at a small fraction of
    another reference series (typically ATR)."""
    return x.mask(x <= 0, floor_frac_of * frac)


# ============================================================================ single-candle
def hammer(g: Geometry, body_atr_max: float = 0.5, wick_body_ratio: float = 2.0,
          opposite_wick_frac: float = 0.3) -> pd.Series:
    """Small body near the top of the range, long lower wick, little/no upper wick, after a downtrend."""
    body_safe = _safe(g.body, g.atr)
    shape = (g.body <= body_atr_max * g.atr) & (g.lower >= wick_body_ratio * body_safe) & \
            (g.upper <= opposite_wick_frac * g.range_.replace(0, np.nan))
    sig = shape & (g.trend < 0)
    return pd.Series(np.where(sig, BULLISH, NONE), index=g.close.index)


def hanging_man(g: Geometry, **kw) -> pd.Series:
    """Same shape as Hammer, but after an uptrend -> bearish reversal warning."""
    shape = _shape_only(hammer, g, **kw)
    return pd.Series(np.where(shape & (g.trend > 0), BEARISH, NONE), index=g.close.index)


def inverted_hammer(g: Geometry, body_atr_max: float = 0.5, wick_body_ratio: float = 2.0,
                    opposite_wick_frac: float = 0.3) -> pd.Series:
    """Small body near the bottom, long upper wick, after a downtrend."""
    body_safe = _safe(g.body, g.atr)
    shape = (g.body <= body_atr_max * g.atr) & (g.upper >= wick_body_ratio * body_safe) & \
            (g.lower <= opposite_wick_frac * g.range_.replace(0, np.nan))
    sig = shape & (g.trend < 0)
    return pd.Series(np.where(sig, BULLISH, NONE), index=g.close.index)


def shooting_star(g: Geometry, **kw) -> pd.Series:
    """Same shape as Inverted Hammer, but after an uptrend -> bearish reversal warning."""
    shape = _shape_only(inverted_hammer, g, **kw)
    return pd.Series(np.where(shape & (g.trend > 0), BEARISH, NONE), index=g.close.index)


def _shape_only(fn, g: Geometry, **kw) -> pd.Series:
    """Re-run a directional pattern's shape test without its trend-direction gate (used to share shape
    logic between trend-context pairs like Hammer/Hanging Man)."""
    neutral = Geometry(g.open, g.high, g.low, g.close, g.body, g.top, g.bot, g.upper, g.lower, g.range_,
                       g.is_bull, g.is_bear, g.atr, pd.Series(-1, index=g.close.index))
    return fn(neutral, **kw) != NONE


def doji(g: Geometry, body_atr_max: float = 0.1) -> pd.Series:
    """Open ~= close. Not directional on its own - direction comes from what precedes/follows it."""
    is_doji = g.body <= body_atr_max * g.atr.replace(0, np.nan)
    return is_doji.fillna(False)


def doji_dragonfly(g: Geometry, body_atr_max: float = 0.1, upper_wick_max_frac: float = 0.1) -> pd.Series:
    """Doji with a long lower wick and (almost) no upper wick - bullish at the bottom of a move."""
    d = doji(g, body_atr_max) & (g.upper <= upper_wick_max_frac * g.range_.replace(0, np.nan)) & (g.lower > 0)
    return pd.Series(np.where(d, BULLISH, NONE), index=g.close.index)


def doji_gravestone(g: Geometry, body_atr_max: float = 0.1, lower_wick_max_frac: float = 0.1) -> pd.Series:
    """Doji with a long upper wick and (almost) no lower wick - bearish at the top of a move."""
    d = doji(g, body_atr_max) & (g.lower <= lower_wick_max_frac * g.range_.replace(0, np.nan)) & (g.upper > 0)
    return pd.Series(np.where(d, BEARISH, NONE), index=g.close.index)


def doji_long_legged(g: Geometry, body_atr_max: float = 0.1, min_wick_atr: float = 0.5) -> pd.Series:
    """Doji with long wicks on both sides - high indecision, no directional bias on its own."""
    d = doji(g, body_atr_max) & (g.upper >= min_wick_atr * g.atr) & (g.lower >= min_wick_atr * g.atr)
    return d.fillna(False)


def marubozu(g: Geometry, body_atr_min: float = 1.2, wick_max_frac: float = 0.05) -> pd.Series:
    """Full-bodied candle, little/no wicks either side - strong conviction in the close's direction."""
    small_wicks = (g.upper <= wick_max_frac * g.range_.replace(0, np.nan)) & \
                  (g.lower <= wick_max_frac * g.range_.replace(0, np.nan))
    strong_body = g.body >= body_atr_min * g.atr
    return pd.Series(np.select([strong_body & small_wicks & g.is_bull, strong_body & small_wicks & g.is_bear],
                               [BULLISH, BEARISH], default=NONE), index=g.close.index)


def spinning_top(g: Geometry, body_atr_max: float = 0.4, wick_min_frac: float = 0.3) -> pd.Series:
    """Small body, both wicks notably larger than the body, roughly balanced - indecision."""
    body_safe = _safe(g.body, g.atr)
    small_body = g.body <= body_atr_max * g.atr
    balanced_wicks = (g.upper >= wick_min_frac * g.range_.replace(0, np.nan)) & \
                     (g.lower >= wick_min_frac * g.range_.replace(0, np.nan))
    long_vs_body = (g.upper >= body_safe) & (g.lower >= body_safe)
    return (small_body & balanced_wicks & long_vs_body).fillna(False)


def belt_hold(g: Geometry, body_atr_min: float = 0.8, wick_max_frac: float = 0.1) -> pd.Series:
    """Opens at (bullish) or near the low with almost no lower wick and closes near the high, or the
    bearish mirror - a full-bodied candle with one side "shaved" flat."""
    bull = g.is_bull & (g.lower <= wick_max_frac * g.range_.replace(0, np.nan)) & (g.body >= body_atr_min * g.atr)
    bear = g.is_bear & (g.upper <= wick_max_frac * g.range_.replace(0, np.nan)) & (g.body >= body_atr_min * g.atr)
    return pd.Series(np.select([bull, bear], [BULLISH, BEARISH], default=NONE), index=g.close.index)


# ============================================================================ two-candle
def _prev(g: Geometry, series: pd.Series) -> pd.Series:
    return series.shift(1)


def engulfing(g: Geometry) -> pd.Series:
    """Candle 2's body fully engulfs candle 1's body, in the opposite colour."""
    p_top, p_bot, p_bull, p_bear = _prev(g, g.top), _prev(g, g.bot), _prev(g, g.is_bull), _prev(g, g.is_bear)
    bull = g.is_bull & p_bear & (g.close >= p_top) & (g.open <= p_bot)
    bear = g.is_bear & p_bull & (g.open >= p_top) & (g.close <= p_bot)
    return pd.Series(np.select([bull, bear], [BULLISH, BEARISH], default=NONE), index=g.close.index)


def harami(g: Geometry) -> pd.Series:
    """Candle 2's body sits entirely inside candle 1's (larger) body, opposite colour - contraction
    after a strong move, possible reversal."""
    p_top, p_bot, p_bull, p_bear, p_body = (_prev(g, g.top), _prev(g, g.bot), _prev(g, g.is_bull),
                                            _prev(g, g.is_bear), _prev(g, g.body))
    inside = (g.top <= p_top) & (g.bot >= p_bot) & (g.body < p_body)
    bull = inside & p_bear & g.is_bull
    bear = inside & p_bull & g.is_bear
    return pd.Series(np.select([bull, bear], [BULLISH, BEARISH], default=NONE), index=g.close.index)


def harami_cross(g: Geometry, body_atr_max: float = 0.1) -> pd.Series:
    """Harami where the second (inside) candle is itself a doji - stronger indecision signal."""
    h = harami(g)
    is_doji_now = doji(g, body_atr_max)
    return pd.Series(np.where(h != NONE, np.sign(h) * 100, NONE), index=g.close.index).where(is_doji_now, NONE)


def piercing_line(g: Geometry, min_penetration: float = 0.5) -> pd.Series:
    """Bullish: candle 2 opens below candle 1's low and closes above the midpoint of candle 1's
    (bearish) body."""
    p_open, p_close, p_bear = _prev(g, g.open), _prev(g, g.close), _prev(g, g.is_bear)
    midpoint = (p_open + p_close) / 2
    sig = p_bear & g.is_bull & (g.open < _prev(g, g.low)) & (g.close > midpoint) & (g.close < p_open)
    return pd.Series(np.where(sig, BULLISH, NONE), index=g.close.index)


def dark_cloud_cover(g: Geometry) -> pd.Series:
    """Bearish mirror of Piercing Line: opens above candle 1's high, closes below the midpoint of
    candle 1's (bullish) body."""
    p_open, p_close, p_bull = _prev(g, g.open), _prev(g, g.close), _prev(g, g.is_bull)
    midpoint = (p_open + p_close) / 2
    sig = p_bull & g.is_bear & (g.open > _prev(g, g.high)) & (g.close < midpoint) & (g.close > p_open)
    return pd.Series(np.where(sig, BEARISH, NONE), index=g.close.index)


def tweezer(g: Geometry, match_tol_atr: float = 0.1) -> pd.Series:
    """Two candles with matching highs (Tweezer Top, after an uptrend) or matching lows (Tweezer
    Bottom, after a downtrend) - a stall right at the same level twice in a row."""
    tol = match_tol_atr * g.atr.replace(0, np.nan)
    top_match = (g.high - _prev(g, g.high)).abs() <= tol
    bot_match = (g.low - _prev(g, g.low)).abs() <= tol
    bear = top_match & (g.trend > 0) & g.is_bear
    bull = bot_match & (g.trend < 0) & g.is_bull
    return pd.Series(np.select([bull, bear], [BULLISH, BEARISH], default=NONE), index=g.close.index)


def kicker(g: Geometry, gap_atr_min: float = 0.3) -> pd.Series:
    """Candle 1 and 2 are opposite-coloured marubozu-like candles with a gap between them and no
    overlap at all - an abrupt sentiment flip."""
    p_open, p_close, p_bull, p_bear = _prev(g, g.open), _prev(g, g.close), _prev(g, g.is_bull), _prev(g, g.is_bear)
    gap = gap_atr_min * g.atr
    bull = p_bear & g.is_bull & (g.open >= p_open + gap) & (g.low >= p_open)
    bear = p_bull & g.is_bear & (g.open <= p_open - gap) & (g.high <= p_open)
    return pd.Series(np.select([bull, bear], [BULLISH, BEARISH], default=NONE), index=g.close.index)


def on_neck(g: Geometry, tol_atr: float = 0.05) -> pd.Series:
    """Bearish continuation: a long down candle followed by a small up candle closing right at (or
    just above) the prior candle's low - the "rescue" fails to make real progress."""
    p_low, p_bear, p_body = _prev(g, g.low), _prev(g, g.is_bear), _prev(g, g.body)
    strong_prev = p_body >= 0.8 * g.atr
    sig = p_bear & strong_prev & g.is_bull & ((g.close - p_low).abs() <= tol_atr * g.atr) & (g.trend < 0)
    return pd.Series(np.where(sig, BEARISH, NONE), index=g.close.index)


def in_neck(g: Geometry, tol_atr: float = 0.05) -> pd.Series:
    """Variant of On-Neck: the small up candle closes just inside the prior down candle's body (near
    its close) rather than exactly at the low - also a weak continuation signal."""
    p_low, p_close, p_bear, p_body = _prev(g, g.low), _prev(g, g.close), _prev(g, g.is_bear), _prev(g, g.body)
    strong_prev = p_body >= 0.8 * g.atr
    sig = (p_bear & strong_prev & g.is_bull & (g.close > p_low) &
          ((g.close - p_close).abs() <= tol_atr * g.atr) & (g.trend < 0))
    return pd.Series(np.where(sig, BEARISH, NONE), index=g.close.index)


def matching_low(g: Geometry, tol_atr: float = 0.05) -> pd.Series:
    """Two bearish candles with (almost) the same close/low - buyers repeatedly defending a level,
    bullish continuation-of-support signal in a downtrend."""
    p_close, p_bear = _prev(g, g.close), _prev(g, g.is_bear)
    sig = p_bear & g.is_bear & ((g.close - p_close).abs() <= tol_atr * g.atr) & (g.trend < 0)
    return pd.Series(np.where(sig, BULLISH, NONE), index=g.close.index)


# ============================================================================ three(+)-candle
def morning_evening_star(g: Geometry, star_body_atr_max: float = 0.5, penetration: float = 0.5) -> pd.Series:
    """Morning Star: big down candle, small-bodied "star" gapping below it, big up candle closing
    back into (at least) the midpoint of the first candle's body. Evening Star is the mirror.
    Covers the plain and doji-star variants (a doji star just makes `star_body_atr_max` easily satisfied).
    """
    c1_open, c1_close, c1_bear, c1_bull = g.open.shift(2), g.close.shift(2), g.is_bear.shift(2), g.is_bull.shift(2)
    c1_body = g.body.shift(2)
    star_top, star_bot = g.top.shift(1), g.bot.shift(1)
    star_small = g.body.shift(1) <= star_body_atr_max * g.atr.shift(1)
    mid1 = (c1_open + c1_close) / 2

    morning = (c1_bear & (c1_body >= 0.6 * g.atr.shift(2)) & star_small & (star_top <= c1_close) &
              g.is_bull & (g.close >= mid1) & (g.close > c1_close))
    evening = (c1_bull & (c1_body >= 0.6 * g.atr.shift(2)) & star_small & (star_bot >= c1_close) &
              g.is_bear & (g.close <= mid1) & (g.close < c1_close))
    return pd.Series(np.select([morning, evening], [BULLISH, BEARISH], default=NONE), index=g.close.index)


def three_soldiers_crows(g: Geometry, min_body_atr: float = 0.5, max_wick_frac: float = 0.3) -> pd.Series:
    """Three White Soldiers: three consecutive solid bullish candles, each opening within the previous
    body and closing higher, with small upper wicks (steady, un-exhausted advance). Three Black Crows
    is the exact bearish mirror."""
    bull3 = g.is_bull & g.is_bull.shift(1) & g.is_bull.shift(2)
    bodies_ok = ((g.body >= min_body_atr * g.atr) & (g.body.shift(1) >= min_body_atr * g.atr.shift(1)) &
                (g.body.shift(2) >= min_body_atr * g.atr.shift(2)))
    rising = (g.close > g.close.shift(1)) & (g.close.shift(1) > g.close.shift(2))
    opens_within = ((g.open > g.open.shift(1)) & (g.open < g.close.shift(1)) &
                   (g.open.shift(1) > g.open.shift(2)) & (g.open.shift(1) < g.close.shift(2)))
    small_upper = ((g.upper <= max_wick_frac * g.range_.replace(0, np.nan)) &
                  (g.upper.shift(1) <= max_wick_frac * g.range_.shift(1).replace(0, np.nan)))
    soldiers = bull3 & bodies_ok & rising & opens_within & small_upper

    bear3 = g.is_bear & g.is_bear.shift(1) & g.is_bear.shift(2)
    falling = (g.close < g.close.shift(1)) & (g.close.shift(1) < g.close.shift(2))
    opens_within_b = ((g.open < g.open.shift(1)) & (g.open > g.close.shift(1)) &
                      (g.open.shift(1) < g.open.shift(2)) & (g.open.shift(1) > g.close.shift(2)))
    small_lower = ((g.lower <= max_wick_frac * g.range_.replace(0, np.nan)) &
                  (g.lower.shift(1) <= max_wick_frac * g.range_.shift(1).replace(0, np.nan)))
    crows = bear3 & bodies_ok & falling & opens_within_b & small_lower
    return pd.Series(np.select([soldiers, crows], [BULLISH, BEARISH], default=NONE), index=g.close.index)


def three_inside(g: Geometry) -> pd.Series:
    """Three Inside Up/Down: a Harami (candles 1-2) followed by a third candle that closes beyond
    candle 1's open/close, confirming the reversal the Harami only hinted at."""
    h = harami(g).shift(1)
    c1_open, c1_close = g.open.shift(2), g.close.shift(2)
    up = (h == BULLISH) & g.is_bull & (g.close > c1_open)
    down = (h == BEARISH) & g.is_bear & (g.close < c1_open)
    return pd.Series(np.select([up, down], [BULLISH, BEARISH], default=NONE), index=g.close.index)


def three_outside(g: Geometry) -> pd.Series:
    """Three Outside Up/Down: an Engulfing (candles 1-2) followed by a third candle that continues in
    the same direction, confirming it."""
    e = engulfing(g).shift(1)
    up = (e == BULLISH) & g.is_bull & (g.close > g.close.shift(1))
    down = (e == BEARISH) & g.is_bear & (g.close < g.close.shift(1))
    return pd.Series(np.select([up, down], [BULLISH, BEARISH], default=NONE), index=g.close.index)


def abandoned_baby(g: Geometry, star_body_atr_max: float = 0.3, gap_atr_min: float = 0.1) -> pd.Series:
    """Like a Morning/Evening Star, but the star candle gaps away from *both* neighbours with no
    overlapping range on either side - rare, and a stronger version of the star patterns."""
    c1_low, c1_high = g.low.shift(2), g.high.shift(2)
    c1_bear, c1_bull = g.is_bear.shift(2), g.is_bull.shift(2)
    star_high, star_low = g.high.shift(1), g.low.shift(1)
    star_small = g.body.shift(1) <= star_body_atr_max * g.atr.shift(1)
    gap = gap_atr_min * g.atr

    bull_gap_down = star_high < c1_low - gap.shift(2)
    bull_gap_up = g.low > star_high + gap
    morning = c1_bear & star_small & bull_gap_down & g.is_bull & bull_gap_up

    bear_gap_up = star_low > c1_high + gap.shift(2)
    bear_gap_down = g.high < star_low - gap
    evening = c1_bull & star_small & bear_gap_up & g.is_bear & bear_gap_down
    return pd.Series(np.select([morning, evening], [BULLISH, BEARISH], default=NONE), index=g.close.index)


def rising_falling_three_methods(g: Geometry, small_body_atr_max: float = 0.6) -> pd.Series:
    """Continuation pattern: a strong candle, three small candles that stay within its range (a brief
    pause), then a strong candle in the original direction breaking to a new extreme - the trend
    "catches its breath" without actually reversing."""
    first_open, first_close = g.open.shift(4), g.close.shift(4)
    first_bull, first_bear = g.is_bull.shift(4), g.is_bear.shift(4)
    first_body = g.body.shift(4)
    first_top, first_bot = g.top.shift(4), g.bot.shift(4)

    small = pd.concat([g.body.shift(k) <= small_body_atr_max * g.atr.shift(k) for k in (1, 2, 3)], axis=1).all(axis=1)
    contained = pd.concat([(g.high.shift(k) <= first_top) & (g.low.shift(k) >= first_bot) for k in (1, 2, 3)],
                          axis=1).all(axis=1)
    middle_bear = pd.concat([g.is_bear.shift(k) for k in (1, 2, 3)], axis=1).all(axis=1)
    middle_bull = pd.concat([g.is_bull.shift(k) for k in (1, 2, 3)], axis=1).all(axis=1)

    rising = (first_bull & (first_body >= 0.8 * g.atr.shift(4)) & small & contained & middle_bear &
             g.is_bull & (g.close > first_close))
    falling = (first_bear & (first_body >= 0.8 * g.atr.shift(4)) & small & contained & middle_bull &
              g.is_bear & (g.close < first_close))
    return pd.Series(np.select([rising, falling], [BULLISH, BEARISH], default=NONE), index=g.close.index)


def tri_star(g: Geometry, body_atr_max: float = 0.1, gap_atr_min: float = 0.05) -> pd.Series:
    """Three consecutive dojis, the middle one gapping away from the other two - very rare, marks
    extreme indecision at a potential turning point."""
    d0, d1, d2 = doji(g, body_atr_max).shift(2), doji(g, body_atr_max).shift(1), doji(g, body_atr_max)
    all_doji = d0 & d1 & d2
    gap = gap_atr_min * g.atr
    mid_top, mid_bot = g.top.shift(1), g.bot.shift(1)
    c0_bot, c2_top = g.bot.shift(2), g.top
    gap_up_then_down = (mid_bot > g.top.shift(2) + gap.shift(2)) & (g.top < mid_bot - gap)
    gap_down_then_up = (mid_top < c0_bot - gap.shift(2)) & (g.bot > mid_top + gap)
    bearish = all_doji & gap_up_then_down    # gapped up into the star, gapped back down -> top reversal
    bullish = all_doji & gap_down_then_up     # gapped down into the star, gapped back up -> bottom reversal
    return pd.Series(np.select([bullish, bearish], [BULLISH, BEARISH], default=NONE), index=g.close.index)


# ============================================================================ registry
PATTERNS = {
    "hammer": hammer, "hanging_man": hanging_man, "inverted_hammer": inverted_hammer,
    "shooting_star": shooting_star, "doji_dragonfly": doji_dragonfly, "doji_gravestone": doji_gravestone,
    "doji_long_legged": doji_long_legged, "marubozu": marubozu, "spinning_top": spinning_top,
    "belt_hold": belt_hold, "engulfing": engulfing, "harami": harami, "harami_cross": harami_cross,
    "piercing_line": piercing_line, "dark_cloud_cover": dark_cloud_cover, "tweezer": tweezer,
    "kicker": kicker, "on_neck": on_neck, "in_neck": in_neck, "matching_low": matching_low,
    "morning_evening_star": morning_evening_star, "three_soldiers_crows": three_soldiers_crows,
    "three_inside": three_inside, "three_outside": three_outside, "abandoned_baby": abandoned_baby,
    "rising_falling_three_methods": rising_falling_three_methods, "tri_star": tri_star,
}


def detect_all(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
              atr: pd.Series | None = None, patterns: dict | None = None) -> pd.DataFrame:
    """Run every registered pattern once, sharing the same precomputed geometry. Boolean-shaped
    patterns (spinning_top, doji_long_legged) are returned as-is (not signed - they have no inherent
    direction); the rest use the +100/0/-100 convention."""
    g = compute_geometry(open_, high, low, close, atr)
    out = {}
    for name, fn in (patterns or PATTERNS).items():
        out[name] = fn(g)
    return pd.DataFrame(out, index=close.index)
