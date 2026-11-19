"""Context scoring: a pattern detection alone is weak evidence - a hammer in the middle of nowhere
means much less than a hammer at support, with a volume spike, agreeing with the higher timeframe, in a
trending regime. This module scores each detection 0-100 from those factors; strategies only act on
detections above a configurable threshold (see backend/strategies).

Every component score is in [0, 1]; the composite is a weighted sum rescaled to [0, 100]. Weights are
plain arguments (not hardcoded) so the evolution loop can tune them per stock/strategy over time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEFAULT_WEIGHTS = {"location": 0.35, "trend": 0.20, "volume": 0.20, "higher_tf": 0.15, "regime": 0.10}


# ---------------------------------------------------------------------- component scores
def location_score(close: pd.Series, direction: pd.Series, atr: pd.Series,
                   nearest_support: pd.Series | None = None, nearest_resistance: pd.Series | None = None,
                   vwap: pd.Series | None = None, prior_day_high: pd.Series | None = None,
                   prior_day_low: pd.Series | None = None, proximity_atr: float = 1.0) -> pd.Series:
    """1.0 if price sits right at a level that supports the pattern's direction (support for a bullish
    signal, resistance for a bearish one - also checks VWAP and the prior day's high/low), decaying
    linearly to 0 at `proximity_atr` * ATR away. Uses whichever reference levels are supplied; if none
    are close enough (or none given), the score is 0 - a pattern with no location context isn't
    penalized to be wrong, just uninformative on this axis, which is what a 0 here means."""
    atr_safe = atr.replace(0, np.nan)
    best = pd.Series(0.0, index=close.index)

    def consider(level: pd.Series | None, applies_when: pd.Series) -> None:
        nonlocal best
        if level is None:
            return
        dist = (close - level).abs() / atr_safe
        s = (1 - dist / proximity_atr).clip(lower=0, upper=1).fillna(0)
        best[applies_when] = np.maximum(best[applies_when], s[applies_when])

    bullish, bearish = direction > 0, direction < 0
    consider(nearest_support, bullish)
    consider(nearest_resistance, bearish)
    consider(prior_day_low, bullish)
    consider(prior_day_high, bearish)
    if vwap is not None:
        consider(vwap, bullish | bearish)   # VWAP acts as support or resistance depending on approach side
    return best


def trend_score(prior_trend: pd.Series, direction: pd.Series) -> pd.Series:
    """1.0 if the pattern is a reversal signal appearing after a move in the opposite direction (a
    bullish pattern after a downtrend, or vice versa - the textbook setup for every reversal pattern in
    candlestick.py), 0.5 if there's no clear prior trend, 0.0 if the pattern fights an existing move in
    its own direction (e.g. a "bullish" signal after already-strong buying, i.e. chasing)."""
    aligned = (direction > 0) & (prior_trend < 0) | (direction < 0) & (prior_trend > 0)
    fighting = (direction > 0) & (prior_trend > 0) | (direction < 0) & (prior_trend < 0)
    return pd.Series(np.select([aligned, fighting], [1.0, 0.0], default=0.5), index=prior_trend.index)


def volume_score(volume: pd.Series, avg_volume: pd.Series, strong_ratio: float = 1.5) -> pd.Series:
    """1.0 once volume reaches `strong_ratio`x the recent average (a real move, not a quiet drift),
    scaling linearly from 0 at 1x average up to 1.0 at `strong_ratio`x, capped at 1.0 above that."""
    ratio = (volume / avg_volume.replace(0, np.nan)).fillna(0)
    return ((ratio - 1.0) / (strong_ratio - 1.0)).clip(lower=0, upper=1)


def higher_tf_score(higher_tf_trend: pd.Series, direction: pd.Series) -> pd.Series:
    """1.0 if the higher-timeframe trend agrees with the pattern's direction, 0.5 if the higher
    timeframe has no clear trend, 0.0 if it disagrees."""
    agree = (direction > 0) & (higher_tf_trend > 0) | (direction < 0) & (higher_tf_trend < 0)
    disagree = (direction > 0) & (higher_tf_trend < 0) | (direction < 0) & (higher_tf_trend > 0)
    return pd.Series(np.select([agree, disagree], [1.0, 0.0], default=0.5), index=higher_tf_trend.index)


def regime_score(regime: pd.Series, direction: pd.Series, favorable: dict[str, set[str]] | None = None
                 ) -> pd.Series:
    """1.0 if the current market regime label is in the set of regimes favorable for this direction,
    0.5 if regime is unknown/missing, else 0.0. Default: reversal patterns (which is everything in
    candlestick.py/chart_patterns.py) are favored in "ranging" or "high_vol" regimes and penalized in a
    strongly trending regime that opposes them - see backend.features.regime for the regime labels."""
    favorable = favorable or {"bull": {"trending_up", "ranging", "high_vol"}, "bear": {"trending_down", "ranging", "high_vol"}}
    bull_ok = regime.isin(favorable["bull"])
    bear_ok = regime.isin(favorable["bear"])
    unknown = regime.isna() | (regime == "") | (regime == "unknown")
    score = pd.Series(0.0, index=regime.index)
    score[(direction > 0) & bull_ok] = 1.0
    score[(direction < 0) & bear_ok] = 1.0
    score[unknown] = 0.5
    return score


# ---------------------------------------------------------------------- composite
@dataclass
class ContextInputs:
    """Everything context scoring needs, pre-aligned to the same index as the pattern signals."""
    close: pd.Series
    atr: pd.Series
    prior_trend: pd.Series
    volume: pd.Series
    avg_volume: pd.Series
    higher_tf_trend: pd.Series | None = None
    regime: pd.Series | None = None
    nearest_support: pd.Series | None = None
    nearest_resistance: pd.Series | None = None
    vwap: pd.Series | None = None
    prior_day_high: pd.Series | None = None
    prior_day_low: pd.Series | None = None


def score_direction(direction: pd.Series, ctx: ContextInputs, weights: dict[str, float] | None = None,
                    proximity_atr: float = 1.0, strong_volume_ratio: float = 1.5) -> pd.DataFrame:
    """Composite 0-100 score for a signed direction Series (+1/-1/0 per bar). Returns a DataFrame with
    each component score plus the weighted `score` column, so callers/dashboards can show the breakdown."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    total_w = sum(w.values())
    loc = location_score(ctx.close, direction, ctx.atr, ctx.nearest_support, ctx.nearest_resistance,
                        ctx.vwap, ctx.prior_day_high, ctx.prior_day_low, proximity_atr)
    trend = trend_score(ctx.prior_trend, direction)
    vol = volume_score(ctx.volume, ctx.avg_volume, strong_volume_ratio)
    htf = (higher_tf_score(ctx.higher_tf_trend, direction) if ctx.higher_tf_trend is not None
          else pd.Series(0.5, index=direction.index))
    reg = regime_score(ctx.regime, direction) if ctx.regime is not None else pd.Series(0.5, index=direction.index)
    composite = (w["location"] * loc + w["trend"] * trend + w["volume"] * vol +
                w["higher_tf"] * htf + w["regime"] * reg) / total_w * 100
    out = pd.DataFrame({"location": loc, "trend": trend, "volume": vol, "higher_tf": htf, "regime": reg,
                       "score": composite})
    out.loc[direction == 0, :] = np.nan
    return out


def score_patterns(patterns: pd.DataFrame, ctx: ContextInputs, weights: dict[str, float] | None = None,
                   min_score: float = 60.0, proximity_atr: float = 1.0,
                   strong_volume_ratio: float = 1.5) -> pd.DataFrame:
    """Score every column of a pattern-detection DataFrame (e.g. `candlestick.detect_all()`'s output).
    Non-directional columns (plain booleans like spinning_top/doji_long_legged) are skipped - they carry
    no direction to score. Returns one `<pattern>_score` column per directional pattern, plus
    `best_pattern` (the highest-scoring pattern name on each bar) and `best_score`."""
    scores = {}
    for col in patterns.columns:
        series = patterns[col]
        if series.dtype == bool:
            continue
        direction = np.sign(series)
        if (direction == 0).all():
            continue
        scores[col] = score_direction(direction, ctx, weights, proximity_atr, strong_volume_ratio)["score"]
    if not scores:
        empty = pd.DataFrame(index=patterns.index)
        empty["best_pattern"], empty["best_score"], empty["actionable"] = None, np.nan, False
        return empty
    score_df = pd.DataFrame(scores)
    best_score = score_df.max(axis=1, skipna=True)
    valid = score_df.notna().any(axis=1)
    best_pattern = pd.Series(None, index=score_df.index, dtype=object)
    if valid.any():   # idxmax raises on an all-NaN row, so only call it where at least one column fired
        best_pattern.loc[valid] = score_df.loc[valid].idxmax(axis=1, skipna=True)
    out = score_df.add_suffix("_score")
    out["best_pattern"] = best_pattern
    out["best_score"] = best_score
    out["actionable"] = best_score >= min_score
    return out
