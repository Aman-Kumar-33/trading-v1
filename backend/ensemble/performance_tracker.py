"""Tracks each strategy's own rolling, regime-conditioned performance and turns it into a 0-1 weight the
allocator uses to filter, resolve conflicts, AND size positions.

What gets recorded is the NET R-multiple of every signal a strategy emits, measured as if taken alone
(see backend/ensemble/shadow_book.py and full_system_backtest's shadow tracking) - not just the trades
the system happened to execute. Measuring only executed trades has two failure modes: a muted strategy
stops generating evidence and can never recover (a death spiral), and when several strategies agree only
the one picked gets credit (selection bias).

Weight = mean net R over the window / `full_weight_r`, clipped to [floor, 1]:
  - risk-adjusted and cost-inclusive: a 30%-win-rate strategy whose winners are 3R outweighs a 60%-win
    strategy that nets +0.02R, which a win-rate weight gets backwards
  - scale-free (R units), so no rupee magnitudes to tune
  - expectancy at or below `min_edge_r` (+0.05R by default, i.e. a margin over costs, not just "not
    losing") -> `floor` (0 by default): effectively muted, but still shadow-tracked so it can come back
  - a regime with fewer than `min_trades_for_weighting` samples falls back to the strategy's record
    pooled across all regimes (long-term strategies fire ~1-2x a year per symbol - waiting for 10
    samples in each of 4 regimes would keep a proven strategy at probation size for years)
  - no usable evidence at all -> `neutral_weight` (0.25): a probation size, smaller than a proven
    winner's, instead of full size for a strategy nobody has evidence about
"""
from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path


class PerformanceTracker:
    def __init__(self, window: int = 30, min_trades_for_weighting: int = 10, neutral_weight: float = 0.25,
                full_weight_r: float = 0.25, floor: float = 0.1, min_edge_r: float = 0.05):
        self.window = window
        self.min_trades_for_weighting = min_trades_for_weighting
        self.neutral_weight = neutral_weight
        self.full_weight_r = full_weight_r
        self.floor = floor
        self.min_edge_r = min_edge_r
        self._history: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=window))

    def record(self, strategy_id: str, regime: str | None, r_net: float) -> None:
        """`r_net` = net P&L (after all costs) / the rupee risk taken at entry (qty x |entry - stop|)."""
        self._history[(strategy_id, regime or "unknown")].append(float(r_net))

    def weight(self, strategy_id: str, regime: str | None) -> float:
        trades = self._history.get((strategy_id, regime or "unknown"))
        if not trades or len(trades) < self.min_trades_for_weighting:
            trades = [r for (sid, _), rs in self._history.items() if sid == strategy_id for r in rs]
            if len(trades) < self.min_trades_for_weighting:
                return self.neutral_weight
        mean_r = sum(trades) / len(trades)
        if mean_r <= self.min_edge_r:
            return self.floor
        return max(self.floor, min(1.0, mean_r / self.full_weight_r))

    def stats(self) -> dict[str, dict]:
        """Per "strategy|regime": sample count, mean net R, current weight - for reports/dashboards."""
        out = {}
        for (sid, regime), trades in self._history.items():
            out[f"{sid}|{regime}"] = {"n": len(trades), "mean_r": sum(trades) / len(trades) if trades else 0.0,
                                      "weight": self.weight(sid, regime)}
        return out

    def save(self, path: Path) -> None:
        payload = {"config": {"window": self.window, "min_trades_for_weighting": self.min_trades_for_weighting,
                              "neutral_weight": self.neutral_weight, "full_weight_r": self.full_weight_r,
                              "floor": self.floor, "min_edge_r": self.min_edge_r},
                   "history": {f"{sid}|{regime}": list(trades) for (sid, regime), trades in self._history.items()}}
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "PerformanceTracker":
        """Restores both config and history, so a restarted live process (or a live process warm-started
        from a backtest's final state) doesn't re-learn from scratch and re-trade known losers at
        probation size."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tracker = cls(**payload["config"])
        for key, trades in payload["history"].items():
            sid, regime = key.split("|", 1)
            for r in trades:
                tracker.record(sid, regime, r)
        return tracker
