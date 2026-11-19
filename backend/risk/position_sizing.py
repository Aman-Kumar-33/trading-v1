"""Risk-based position sizing: size every trade so a stop-loss hit costs a fixed % of capital, not a
fixed rupee amount per trade (the backtester's `capital_per_trade // entry_price` model, which was fine
for comparing strategies on equal footing but doesn't actually cap downside per trade). This is what a
real risk manager must do instead.
"""
from __future__ import annotations


def risk_based_qty(capital: float, risk_pct: float, entry: float, stop_loss: float, lot_size: int = 1) -> int:
    """Quantity such that a stop-loss hit loses ~`risk_pct` of `capital`. Rounded down to whole lots.
    Returns 0 (never raises) for a degenerate input - zero risk distance, zero/negative capital, or an
    invalid lot size - since "can't size this safely" should mean "don't trade", not a crash."""
    if capital <= 0 or lot_size <= 0:
        return 0
    risk_per_share = abs(entry - stop_loss)
    if risk_per_share <= 0:
        return 0
    qty = int((capital * risk_pct) / risk_per_share)
    return (qty // lot_size) * lot_size
