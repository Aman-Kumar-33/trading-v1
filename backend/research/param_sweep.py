"""Parameter-grid sweep: try every combination in a parameter grid against a shared pool of already-
built features, and pick parameters from a stable plateau rather than a single lucky peak (the plan's
overfitting guard, in its simplest form).

Reuses the exact same build_strategy(overrides=...) -> strategy.run() -> simulate_signals() path the
main backtest and live trading use - a sweep is just many parameter choices run through the same code,
never a separate simulation that could drift out of sync.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import pandas as pd

from backend.risk.risk_manager import RiskLimits
from backend.research.simple_backtester import BacktestResult, simulate_signals
from backend.strategies.base import Strategy, StrategyContext
from backend.strategies.registry import build_strategy

ParamGrid = dict[str, list[Any]]


@dataclass
class SweepCandidate:
    params: dict[str, Any]
    summary: dict[str, Any]

    @property
    def trades(self) -> int:
        return self.summary.get("trades", 0)

    @property
    def expectancy(self) -> float:
        return self.summary.get("expectancy", float("-inf"))


def combos(grid: ParamGrid) -> list[dict[str, Any]]:
    """Every combination in the grid, as a list of override dicts. An empty grid gives one no-op combo."""
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def evaluate_combo(cls: type[Strategy], version: str, overrides: dict[str, Any],
                   feature_by_symbol: dict[str, pd.DataFrame], timeframe: str,
                   capital_per_trade: float, entry_fill_mode: str = "next_open",
                   limits: RiskLimits | None = None) -> SweepCandidate:
    strat = build_strategy(cls, version, overrides=overrides)
    trades = []
    for symbol, feat in feature_by_symbol.items():
        if feat.empty:
            continue
        signals = strat.run(StrategyContext(symbol, timeframe, feat))
        trades.extend(simulate_signals(feat, signals, capital_per_trade=capital_per_trade,
                                       entry_fill_mode=entry_fill_mode, limits=limits).trades)
    return SweepCandidate(params=overrides, summary=BacktestResult(trades=trades).summary())


def sweep(cls: type[Strategy], version: str, grid: ParamGrid, feature_by_symbol: dict[str, pd.DataFrame],
         timeframe: str, capital_per_trade: float = 200_000.0, entry_fill_mode: str = "next_open",
         limits: RiskLimits | None = None) -> list[SweepCandidate]:
    """`capital_per_trade` defaults to Rs 2,00,000 - roughly the per-position cap the risk manager allows
    on a Rs 10L account. At Rs 20,000 (the old default) Angel One's Rs 20/order brokerage cap alone is
    0.1% per leg, which made fees, not edge, decide every intraday result."""
    return [evaluate_combo(cls, version, combo, feature_by_symbol, timeframe, capital_per_trade, entry_fill_mode, limits)
           for combo in combos(grid)]


def _neighbors(params: dict[str, Any], grid: ParamGrid) -> list[dict[str, Any]]:
    """Combos that differ from `params` in exactly one dimension, one grid-step away in either
    direction - used to check the chosen combo sits on a stable plateau, not an isolated spike."""
    out = []
    for key, values in grid.items():
        idx = values.index(params[key])
        for step in (-1, 1):
            j = idx + step
            if 0 <= j < len(values):
                neighbor = dict(params)
                neighbor[key] = values[j]
                out.append(neighbor)
    return out


def _hashable(value: Any) -> Any:
    """Grid values aren't always scalars (e.g. target_r_multiples is a list of lists) - make anything
    hashable so it can key a dict, recursively, without changing its equality semantics."""
    if isinstance(value, list):
        return tuple(_hashable(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    return value


def _param_key(params: dict[str, Any]) -> tuple:
    return tuple(sorted((k, _hashable(v)) for k, v in params.items()))


def plateau_fraction(best: SweepCandidate, all_candidates: list[SweepCandidate], grid: ParamGrid) -> float:
    """Fraction of `best`'s immediate grid-neighbors that are ALSO net-profitable - a cheap proxy for
    "stable plateau, not a lucky peak". 1.0 if `best` has no neighbors to compare against (grid of 1)."""
    by_params = {_param_key(c.params): c for c in all_candidates}
    neighbors = _neighbors(best.params, grid)
    if not neighbors:
        return 1.0
    profitable = sum(
        1 for n in neighbors
        if (cand := by_params.get(_param_key(n))) is not None
        and cand.trades > 0 and cand.summary.get("net_pnl", 0) > 0
    )
    return profitable / len(neighbors)


def pick_best(candidates: list[SweepCandidate], min_trades: int = 30) -> SweepCandidate | None:
    """Highest expectancy among combos with enough trades to be a meaningful sample. Returns None if no
    combo cleared the trade-count floor - callers must not silently fall back to an unreliable result."""
    eligible = [c for c in candidates if c.trades >= min_trades]
    if not eligible:
        return None
    return max(eligible, key=lambda c: c.expectancy)
