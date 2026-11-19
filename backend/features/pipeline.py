"""Feature pipeline: raw clean OHLCV -> one enriched DataFrame with every indicator, pattern, price-action
structure and regime column a strategy needs. Every strategy and the Phase 3 backtester call this so
they all see the exact same features computed the exact same way.

Nothing here looks ahead: every column at row i is computable from rows <= i (verified in
tests/test_pipeline.py), so the same function is correct for live use (call once per new closed candle)
and for backtesting (call once over the whole history).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.features import indicators as ind
from backend.features import pairs as pr
from backend.features import price_action as pa
from backend.features import regime as rg
from backend.features.patterns import candlestick as cdl
from backend.features.patterns import chart_patterns as ch
from backend.core.timeframes import is_intraday

SWING_LEFT_RIGHT = 2


def build_features(raw: pd.DataFrame, timeframe: str, htf_trend: pd.Series | None = None,
                   avg_volume_window: int = 20, daily: pd.DataFrame | None = None,
                   core_only: bool = False) -> pd.DataFrame:
    """raw: columns ts, open, high, low, close, volume (+ optionally close_unadj/day_ok from the cleaner
    - passed through untouched if present). `htf_trend` is an optional higher-timeframe trend Series
    already aligned/reindexed to `raw`'s index (see `align_higher_timeframe` below). `daily` (clean daily
    bars for the same symbol) adds a `daily_trend` column to intraday frames - see `daily_trend`.
    `core_only` skips candlestick/chart patterns and price-action structure (the expensive ~90% of the
    work) - for live use when every strategy on the timeframe only reads core indicators; the core
    columns are computed by exactly the same code either way."""
    intraday = is_intraday(timeframe)
    df = ind.add_core_indicators(raw, intraday=intraday)
    if core_only:
        return _finish(df, avg_volume_window, htf_trend, daily, intraday)

    patterns = cdl.detect_all(df["open"], df["high"], df["low"], df["close"], atr=df["atr_14"])
    df = pd.concat([df, patterns.add_prefix("cdl_")], axis=1)

    sw = pa.swings(df["high"], df["low"], left=SWING_LEFT_RIGHT, right=SWING_LEFT_RIGHT)
    hi_conf = pa.confirmed_at(sw["swing_high"], SWING_LEFT_RIGHT)
    lo_conf = pa.confirmed_at(sw["swing_low"], SWING_LEFT_RIGHT)
    hi_price_conf = sw["swing_high_price"].shift(SWING_LEFT_RIGHT)
    lo_price_conf = sw["swing_low_price"].shift(SWING_LEFT_RIGHT)
    df["swing_high_confirmed"], df["swing_low_confirmed"] = hi_conf, lo_conf
    df["swing_high_price_confirmed"], df["swing_low_price_confirmed"] = hi_price_conf, lo_price_conf

    levels = pa.rolling_nearest_levels(df["close"], hi_conf, lo_conf, hi_price_conf, lo_price_conf)
    df["nearest_support"], df["nearest_resistance"] = levels["nearest_support"], levels["nearest_resistance"]

    struct = pa.analyze_structure(df["open"], df["high"], df["low"], df["close"], left=SWING_LEFT_RIGHT,
                                  right=SWING_LEFT_RIGHT, atr=df["atr_14"])
    df = pd.concat([df, struct.add_prefix("struct_")], axis=1)

    io = pa.inside_outside_bars(df["high"], df["low"])
    df["inside_bar"], df["outside_bar"] = io["inside_bar"], io["outside_bar"]
    df["pin_bar"] = pa.pin_bar(df["open"], df["high"], df["low"], df["close"])
    fvg = pa.fair_value_gaps(df["high"], df["low"])
    df["bullish_fvg"], df["bearish_fvg"] = fvg["bullish_fvg"], fvg["bearish_fvg"]

    pivots = ch.zigzag_pivots(hi_conf, lo_conf, hi_price_conf, lo_price_conf)
    for name, fn in (("double", ch.double_top_bottom), ("triple", ch.triple_top_bottom),
                     ("head_shoulders", ch.head_shoulders), ("rectangle", ch.rectangle)):
        df[f"chart_{name}"] = fn(pivots, df["close"], df["atr_14"], df.index)
    twc = ch.triangles_wedges_channels(pivots, df["close"], df["atr_14"], df.index)
    df["chart_pattern"], df["chart_signal"] = twc["pattern"], twc["signal"]
    df["chart_flag_pennant"] = ch.flag_pennant(df["open"], df["high"], df["low"], df["close"], df["atr_14"])
    df["chart_cup_handle"] = ch.cup_handle(df["close"], df["high"], df["low"], df["atr_14"])
    return _finish(df, avg_volume_window, htf_trend, daily, intraday)


def _finish(df: pd.DataFrame, avg_volume_window: int, htf_trend: pd.Series | None, daily: pd.DataFrame | None,
            intraday: bool) -> pd.DataFrame:
    a = ind.adx(df["high"], df["low"], df["close"], 14)
    atr_pct = rg.atr_percentile(df["atr_14"], 100)
    df["regime"] = rg.classify_regime(a["adx"], a["plus_di"], a["minus_di"], atr_pct)

    df["avg_volume"] = df["volume"].rolling(avg_volume_window, min_periods=max(5, avg_volume_window // 4)).mean()
    df["prior_trend"] = cdl.prior_trend(df["close"])
    if htf_trend is not None:
        df["htf_trend"] = htf_trend.reindex(df.index)
    if daily is not None and intraday:
        df["daily_trend"] = daily_trend(df["ts"], daily)
    return df


DAILY_TREND_LOOKBACK = 200


def daily_trend(intraday_ts: pd.Series, daily: pd.DataFrame, lookback: int = DAILY_TREND_LOOKBACK) -> pd.Series:
    """+1 / -1 / NaN per intraday bar: was the last COMPLETED day's close above or below its `lookback`-day
    moving average (Brock-Lakonishok-LeBaron 1992; Moskowitz-Ooi-Pedersen 2012: trade with the slow trend).
    A bar on day D only ever sees days before D - never today's still-forming daily bar."""
    d = daily[["ts", "close"]].dropna().sort_values("ts")
    diff = (d["close"] - d["close"].rolling(lookback, min_periods=lookback).mean()).to_numpy()
    sign = np.where(np.isnan(diff), np.nan, np.where(diff > 0, 1.0, -1.0))
    day_keys = _naive_days(d["ts"])
    pos = day_keys.searchsorted(_naive_days(intraday_ts), side="left") - 1   # strictly earlier days only
    out = np.full(len(intraday_ts), np.nan)
    valid = pos >= 0
    out[valid] = sign[pos[valid]]
    return pd.Series(out, index=intraday_ts.index)


def _naive_days(ts: pd.Series) -> np.ndarray:
    t = pd.to_datetime(ts)
    if t.dt.tz is not None:
        t = t.dt.tz_localize(None)
    return t.dt.normalize().to_numpy()


def align_higher_timeframe(higher_df: pd.DataFrame, lower_index: pd.Index, higher_ts: pd.Series,
                          lower_ts: pd.Series, trend_col: str = "supertrend_dir") -> pd.Series:
    """Broadcast a higher-timeframe trend column down to a lower timeframe's bars: each lower-timeframe
    bar gets the most recent *closed* higher-timeframe bar's value (never the still-forming one it
    belongs to - that would be lookahead)."""
    ht = pd.Series(higher_df[trend_col].to_numpy(), index=pd.DatetimeIndex(higher_ts)).sort_index()
    pos = ht.index.searchsorted(pd.DatetimeIndex(lower_ts), side="right") - 1
    out = pd.Series(index=lower_index, dtype="float64")
    valid = pos >= 0
    out.iloc[valid] = ht.iloc[pos[valid]].to_numpy()
    return out


def pattern_columns(df: pd.DataFrame) -> pd.DataFrame:
    """All candlestick pattern columns (the ones added with the cdl_ prefix), for convenience."""
    return df[[c for c in df.columns if c.startswith("cdl_")]]
