"""Indian equity cash-market cost model. Rates verified against Angel One's own published rate card
(https://www.angelone.in/exchange-transaction-charges, re-checked 2026-10-01) - STT/GST/stamp duty/SEBI
fee are government/exchange-set; brokerage and DP charges are Angel One's.

Brokerage is the same for intraday and delivery (lower of Rs 20 or 0.1%, min Rs 5 per executed order) -
third-party sites quoting "0.03% intraday" or "Rs 0 delivery" are out of date. Because it's capped per
ORDER, ticket size matters enormously: at a Rs 20,000 ticket it's 0.1% per leg, at Rs 2,00,000 it's
0.01%. Percentage costs (STT, exchange, stamp, GST) don't shrink with size - which is why very tight
stops can't survive Indian intraday fees at any position size.

DP charge (Rs 20 + 18% GST per scrip, every delivery SELL - https://www.angelone.in/support/charges-and-
cashbacks/dp-depository-participant-charges) is per-trade and IS modeled. Not modeled: the temporary
"free brokerage up to Rs 500" new-user promo (a backtest should reflect steady-state costs), annual AMC
(not per-trade), IPFT (Rs ~0.01/crore - negligible), F&O rates (equity cash only).
"""
from __future__ import annotations

from dataclasses import dataclass

from backend.core.models import Horizon, Side

# Angel One standard rates (both delivery/CNC and intraday/MIS), as published 2026-09-29.
BROKERAGE_PCT = 0.001          # 0.1% per executed order...
BROKERAGE_CAP = 20.0           # ...capped at Rs 20...
BROKERAGE_MIN = 5.0            # ...floored at Rs 5
EXCHANGE_TXN_PCT = 0.000030699  # NSE, both delivery and intraday, both sides
SEBI_FEE_PCT = 10 / 1e7          # "Rs 10 per crore of turnover" = 10 / 1,00,00,000 = 0.0001%
GST_RATE = 0.18                  # on (brokerage + exchange txn charge + SEBI fee) only

STT_DELIVERY_PCT = 0.001        # both buy and sell
STT_INTRADAY_PCT = 0.00025      # sell side only
STAMP_DUTY_DELIVERY_PCT = 0.00015   # buy side only
STAMP_DUTY_INTRADAY_PCT = 0.00003   # buy side only
DP_CHARGE = 20.0 * (1 + GST_RATE)   # per scrip, on every delivery sell (not intraday)


@dataclass
class CostBreakdown:
    brokerage: float
    stt: float
    exchange_txn: float
    sebi_fee: float
    gst: float
    stamp_duty: float
    dp_charge: float = 0.0

    @property
    def total(self) -> float:
        return (self.brokerage + self.stt + self.exchange_txn + self.sebi_fee + self.gst + self.stamp_duty
                + self.dp_charge)


def _brokerage(turnover: float) -> float:
    if turnover <= 0:
        return 0.0
    return min(BROKERAGE_CAP, max(BROKERAGE_MIN, turnover * BROKERAGE_PCT))


def leg_cost(price: float, qty: int, side: Side, horizon: Horizon) -> CostBreakdown:
    """Costs for one leg (either the entry or the exit) of a trade."""
    turnover = price * qty
    intraday = horizon == Horizon.INTRADAY
    brokerage = _brokerage(turnover)
    exchange_txn = turnover * EXCHANGE_TXN_PCT
    sebi_fee = turnover * SEBI_FEE_PCT
    gst = (brokerage + exchange_txn + sebi_fee) * GST_RATE

    if intraday:
        stt = turnover * STT_INTRADAY_PCT if side == Side.SELL else 0.0
        stamp_duty = turnover * STAMP_DUTY_INTRADAY_PCT if side == Side.BUY else 0.0
    else:
        stt = turnover * STT_DELIVERY_PCT
        stamp_duty = turnover * STAMP_DUTY_DELIVERY_PCT if side == Side.BUY else 0.0
    dp_charge = DP_CHARGE if (not intraday and side == Side.SELL and turnover > 0) else 0.0

    return CostBreakdown(brokerage=brokerage, stt=stt, exchange_txn=exchange_txn, sebi_fee=sebi_fee,
                         gst=gst, stamp_duty=stamp_duty, dp_charge=dp_charge)


def round_trip_cost(entry_price: float, exit_price: float, qty: int, entry_side: Side, horizon: Horizon
                    ) -> CostBreakdown:
    """Total cost of opening and closing a position - entry leg + the opposite-side exit leg."""
    exit_side = Side.SELL if entry_side == Side.BUY else Side.BUY
    entry = leg_cost(entry_price, qty, entry_side, horizon)
    exit_ = leg_cost(exit_price, qty, exit_side, horizon)
    return CostBreakdown(
        brokerage=entry.brokerage + exit_.brokerage, stt=entry.stt + exit_.stt,
        exchange_txn=entry.exchange_txn + exit_.exchange_txn, sebi_fee=entry.sebi_fee + exit_.sebi_fee,
        gst=entry.gst + exit_.gst, stamp_duty=entry.stamp_duty + exit_.stamp_duty,
        dp_charge=entry.dp_charge + exit_.dp_charge)


def actual_trade_cost(entry_price: float, exit_price: float, qty: int, entry_side: Side, planned: Horizon,
                      overnight: bool) -> CostBreakdown:
    """Costs as the exchange actually charges them: STT/stamp/DP depend on whether shares were DELIVERED,
    not on the strategy's intent. A same-day buy+sell is intraday even on a CNC order; an intraday long
    carried overnight becomes a delivery trade (STT 0.1% both sides, delivery stamp duty, DP on the sell)."""
    horizon = Horizon.LONG_TERM if (overnight or planned == Horizon.LONG_TERM) else Horizon.INTRADAY
    return round_trip_cost(entry_price, exit_price, qty, entry_side, horizon)


def slippage(price: float, atr: float, qty: int, atr_frac: float = 0.05) -> float:
    """A simple liquidity-agnostic slippage estimate: a small fraction of ATR per share, both ways.
    Phase 3's full build-out replaces this with a real liquidity-based model (order book depth /
    average volume) - this is the minimal version needed to avoid backtesting on impossible fills."""
    return atr * atr_frac * qty
