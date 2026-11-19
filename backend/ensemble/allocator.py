"""Meta-allocator: the layer between "what every strategy thinks should happen" and "what the risk
manager gets asked to size". Two jobs, per the plan: resolve multiple strategies disagreeing about the
same symbol at the same moment, and weight surviving signals by each strategy's own recent
regime-conditioned performance - a strategy that's currently losing fades toward zero influence
automatically, without anyone having to notice and disable it by hand.

The 80/20 intraday/long-term capital split lives in backend.risk.risk_manager (RiskLimits) instead of
here - that's where capital actually gets turned into a quantity, so the split belongs there, not
duplicated in both places.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from backend.core.models import Side, Signal
from backend.ensemble.performance_tracker import PerformanceTracker

log = logging.getLogger(__name__)


@dataclass
class AllocatedSignal:
    signal: Signal
    weight: float   # 0-1, combined conviction after merging agreeing strategies - scales position size


class Allocator:
    def __init__(self, tracker: PerformanceTracker | None = None, min_weight: float = 0.05,
                dominance_ratio: float = 1.5):
        self.tracker = tracker or PerformanceTracker()
        self.min_weight = min_weight        # below this, a signal is dropped outright, not just faded
        self.dominance_ratio = dominance_ratio   # how much stronger one side must be to win a conflict
        self.last_drop_reason: str | None = None   # "muted" | "conflict" when decide() returns []

    def decide(self, signals: list[Signal]) -> list[AllocatedSignal]:
        """`signals` must be every ACCEPTED signal for ONE symbol at ONE moment (across strategies) -
        callers group by (symbol, ts) before calling this; mixing symbols/timestamps here is meaningless.
        Returns 0 or 1 AllocatedSignal: either strategies net out to a clear direction, or they don't.
        The returned weight is meant to scale position size (RiskManager.evaluate's `weight`)."""
        self.last_drop_reason = None
        if not signals:
            return []
        weighted = [(s, self.tracker.weight(s.strategy_id, s.regime)) for s in signals]
        weighted = [(s, w) for s, w in weighted if w >= self.min_weight]
        if not weighted:
            self.last_drop_reason = "muted"
            log.info("signal dropped by allocator", extra={"data": {
                "symbol": signals[0].symbol, "strategies": [s.strategy_id for s in signals], "reason": "muted"}})
            return []

        buys = [(s, w) for s, w in weighted if s.side == Side.BUY]
        sells = [(s, w) for s, w in weighted if s.side == Side.SELL]
        buy_w, sell_w = sum(w for _, w in buys), sum(w for _, w in sells)
        if buys and sells:
            if buy_w > sell_w * self.dominance_ratio:
                weighted = buys
            elif sell_w > buy_w * self.dominance_ratio:
                weighted = sells
            else:
                self.last_drop_reason = "conflict"   # no clear majority - stand aside rather than guess
                log.info("signal dropped by allocator", extra={"data": {
                    "symbol": signals[0].symbol, "strategies": [s.strategy_id for s in signals], "reason": "conflict"}})
                return []

        best_signal, _ = max(weighted, key=lambda sw: sw[1])
        combined_weight = min(1.0, sum(w for _, w in weighted))
        if len(weighted) > 1:
            log.info("signals merged", extra={"data": {"symbol": best_signal.symbol,
                     "strategies": [s.strategy_id for s, _ in weighted], "weight": round(combined_weight, 3)}})
        return [AllocatedSignal(signal=best_signal, weight=combined_weight)]
