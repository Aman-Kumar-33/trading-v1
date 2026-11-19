"""Strategy base class.

A strategy takes a feature-enriched DataFrame (backend.features.pipeline.build_features) for one symbol
and emits zero or more Signals. Strategies are *vectorized*: `generate_signals` scans the whole frame in
one call and returns every signal it finds - the same code path backtests years of history and (called
again each time a new candle closes, only ever seeing the new tail) drives live trading.

`Strategy.run()` wraps `generate_signals()` and enforces the one non-negotiable rule from the plan:
every signal must carry a stop loss, or it's rejected before it goes anywhere near the risk manager.
"""
from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from backend.core.models import Horizon, Signal

log = logging.getLogger(__name__)


@dataclass
class StrategyContext:
    symbol: str
    timeframe: str
    df: pd.DataFrame                          # output of backend.features.pipeline.build_features
    extra: dict[str, Any] = field(default_factory=dict)   # e.g. {"pair": (partner_symbol, partner_df)}


class Strategy(ABC):
    strategy_id: str = "base"
    horizon: Horizon = Horizon.INTRADAY
    default_params: dict[str, Any] = {}
    # False = reads only core indicators/regime/volume, so live can use build_features(core_only=True)
    needs_full_features: bool = True

    def __init__(self, params: dict[str, Any] | None = None, version: str = "v1"):
        self.params = {**self.default_params, **(params or {})}
        self.version = version

    @abstractmethod
    def generate_signals(self, ctx: StrategyContext) -> list[Signal]:
        """Return every signal this strategy fires across `ctx.df` (backtest use) or just the newest
        bar(s) (live use - callers typically only act on a signal timestamped at the latest closed
        candle). Every Signal must set `stop_loss`; `run()` rejects any that don't."""

    def run(self, ctx: StrategyContext) -> list[Signal]:
        signals = self.generate_signals(ctx)
        allowed_regimes = self.params.get("allowed_regimes")
        min_confidence = self.params.get("min_confidence")
        # Trade only with the slow daily trend (needs build_features(..., daily=...)). If the column
        # isn't there the filter can't apply - logged once, never silently pretended.
        trend_by_ts = None
        if self.params.get("daily_trend_filter"):
            if "daily_trend" in ctx.df.columns:
                trend_by_ts = dict(zip(ctx.df["ts"], ctx.df["daily_trend"]))
            else:
                log.warning("daily_trend_filter on but no daily_trend column - filter not applied",
                            extra={"data": {"strategy": self.strategy_id, "symbol": ctx.symbol}})
        # Only act on unusually heavy volume (bar volume / its rolling average). In the 2026-10-02 edge audit
        # this was the one condition with a gross edge in both discovery and holdout periods.
        vol_ratio_by_ts = None
        min_volume_ratio = self.params.get("min_volume_ratio")
        if min_volume_ratio and {"volume", "avg_volume"} <= set(ctx.df.columns):
            vol_ratio_by_ts = dict(zip(ctx.df["ts"], ctx.df["volume"] / ctx.df["avg_volume"]))
        accepted: list[Signal] = []
        for sig in signals:
            sig.strategy_id = self.strategy_id
            sig.strategy_version = self.version
            sig.horizon = self.horizon
            if sig.stop_loss is None or not math.isfinite(sig.stop_loss):
                sig.status, sig.reject_reason = "REJECTED", "missing or invalid stop_loss"
                log.warning("signal rejected - no stop loss", extra={"data": {
                    "strategy": self.strategy_id, "symbol": sig.symbol, "ts": str(sig.ts)}})
                continue
            # Both gates are opt-in via config (`allowed_regimes`, `min_confidence`) so strategies/tests
            # that don't set them are completely unaffected - a strategy only gets gated once its v1.yaml
            # actually declares which regime(s) it's meant to trade in / what score it needs.
            if allowed_regimes and sig.regime is not None and sig.regime not in allowed_regimes:
                sig.status, sig.reject_reason = "REJECTED", f"regime {sig.regime!r} not in {allowed_regimes}"
                continue
            if min_confidence is not None and sig.confidence is not None and sig.confidence < min_confidence:
                sig.status, sig.reject_reason = "REJECTED", f"confidence {sig.confidence} < {min_confidence}"
                continue
            if vol_ratio_by_ts is not None:
                ratio = vol_ratio_by_ts.get(sig.ts)
                if ratio is None or not math.isfinite(ratio) or ratio < min_volume_ratio:
                    sig.status, sig.reject_reason = "REJECTED", f"volume ratio below {min_volume_ratio}"
                    continue
            if trend_by_ts is not None:
                trend = trend_by_ts.get(sig.ts)
                wanted = 1.0 if sig.side.value == "BUY" else -1.0
                if trend is None or math.isnan(trend) or trend != wanted:
                    sig.status, sig.reject_reason = "REJECTED", "against (or no) daily trend"
                    continue
            if sig.status == "NEW":
                sig.status = "ACCEPTED"
            accepted.append(sig)
        return accepted

    def config_hash(self) -> str:
        """Stable hash of this strategy's tunable parameters, used to key `strategy_vault/` snapshots."""
        import hashlib
        import json
        blob = json.dumps(self.params, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------- shared helpers for subclasses
def atr_stop(entry: float, atr: float, side: str, mult: float = 1.5) -> float:
    """A stop `mult` ATRs beyond the entry, in the direction that makes sense for the position's side."""
    return entry - mult * atr if side == "BUY" else entry + mult * atr


def r_multiple_targets(entry: float, stop_loss: float, multiples: tuple[float, ...] = (1.0, 2.0)) -> list[float]:
    """Targets at fixed multiples of the initial risk (entry-to-stop distance) - "1R", "2R", ..."""
    risk = entry - stop_loss   # positive for a long (stop below), negative for a short (stop above)
    return [entry + risk * m for m in multiples]
