"""How much of the capital a single MACD buy gets. The system decides per signal (5 / 10 / 15 / 25 % of
capital) instead of using a flat amount:

  - the cross: how fast the MACD histogram turned positive, relative to the stock's ATR (a clean, steep cross
    scores higher than one that barely touches the signal line),
  - the stock's trending score from the daily picker (higher = more MACD crosses / profit lately),
  - volatility: a high ATR% shrinks the size so each trade risks a similar amount.

The tiers are thresholds on a 0-1 conviction score. Same inputs always give the same size, and the reason is
returned as text so it can be logged and shown in Telegram. The 25% ceiling is hard: nothing here, including cash
adjustments, can exceed it.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import floor

TIERS = ((0.80, 0.25), (0.55, 0.15), (0.25, 0.10), (0.0, 0.05))   # (conviction >=, share of capital)
HIST_JUMP_FULL_ATR = 0.06     # histogram rising this many ATRs in one bar = a maximally steep cross
ATR_PCT_LOW, ATR_PCT_SPAN = 0.002, 0.006   # ATR/price of 0.2% scores 1.0 for volatility, 0.8% scores 0.0
WEIGHTS = (0.5, 0.3, 0.2)     # cross strength, trending score, volatility
MAX_SLOT_DIVISOR = 3          # never hold back more than a third of the free cash for the slots still open


@dataclass(frozen=True)
class SizeContext:
    """What the strategy runner knows about a signal when it fires. Any field can be None (carried-over signals)."""
    hist_jump: float | None = None      # macd_hist[now] - macd_hist[previous bar]
    atr: float | None = None
    trend_score: float | None = None    # 0-100 from backend.data.trending, None when the stock list is manual


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))


def conviction(ctx: SizeContext | None, price: float) -> tuple[float | None, str]:
    """(0-1 score, short explanation) - (None, ...) when there is nothing to judge the signal by."""
    if ctx is None or not ctx.atr or ctx.atr <= 0 or ctx.hist_jump is None or price <= 0:
        return None, "no cross data"
    strength = _clip(ctx.hist_jump / (HIST_JUMP_FULL_ATR * ctx.atr))
    trend = _clip(ctx.trend_score / 100.0) if ctx.trend_score is not None else 0.5
    calm = _clip(1.0 - (ctx.atr / price - ATR_PCT_LOW) / ATR_PCT_SPAN)
    score = WEIGHTS[0] * strength + WEIGHTS[1] * trend + WEIGHTS[2] * calm
    label = lambda v: "high" if v >= 0.67 else "moderate" if v >= 0.34 else "weak"   # noqa: E731
    why = (f"{label(strength)} cross, {'' if ctx.trend_score is not None else 'no '}trend score"
           f"{'' if ctx.trend_score is None else f' {ctx.trend_score:.0f}'}, "
           f"{'low' if calm >= 0.67 else 'medium' if calm >= 0.34 else 'high'} volatility")
    return score, why


def choose_pct(ctx: SizeContext | None, price: float, default_pct: float, min_pct: float, max_pct: float) -> tuple[float, str]:
    score, why = conviction(ctx, price)
    if score is None:
        pct, why = default_pct, f"{why} - default size"
    else:
        pct = next(p for floor_, p in TIERS if score >= floor_)
        why = f"conviction {score:.2f} ({why})"
    return max(min_pct, min(max_pct, pct)), why


@dataclass(frozen=True)
class SizePlan:
    qty: int
    budget: float
    pct_of_capital: float
    skip_reason: str | None = None


def plan_size(price: float, capital: float, free_cash: float, open_count: int, max_open: int, pct: float,
              min_pct: float, max_pct: float) -> SizePlan:
    """Quantity for a buy of `pct` of capital, cut down by the 25% ceiling and by the cash that is actually free
    (keeping some back for the slots still open). Below the minimum size, or under one share, it skips."""
    if price <= 0 or capital <= 0:
        return SizePlan(0, 0.0, 0.0, "no valid price / capital")
    if free_cash < min_pct * capital:
        return SizePlan(0, 0.0, 0.0, f"not enough free cash ({free_cash:,.0f} is under the {min_pct:.0%} minimum trade)")
    slots_left = max(1, max_open - open_count) if max_open else 1       # max_open 0 = no limit: nothing is held back for later slots
    budget = min(pct * capital, max_pct * capital, free_cash / min(slots_left, MAX_SLOT_DIVISOR))
    budget = max(budget, min(min_pct * capital, free_cash))   # cash reasons never push a trade under the minimum
    qty = floor(budget / price)
    if qty < 1:
        return SizePlan(0, budget, 0.0, f"one share ({price:,.2f}) costs more than the {budget:,.0f} budget")
    return SizePlan(qty, budget, qty * price / capital)
