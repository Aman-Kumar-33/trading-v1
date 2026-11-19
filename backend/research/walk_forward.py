"""Walk-forward validation: pick parameters on a rolling *train* window via param_sweep, then measure
those exact parameters out-of-sample on the following *test* window. Only OOS results count towards
whether a strategy actually works - this is what tells "tuned to noise" apart from "a real edge".

Folds slice an already-built, full-history feature frame by date rather than rebuilding indicators from
scratch per fold: indicators (EMA/RSI/ATR/...) are then always warmed up from real prior history exactly
as they would be live, instead of cold-starting at each fold's first bar. This is a deliberate
simplification for this first version: like the single-run backtest's own EDGE_BUFFER_BARS finding
(backend/research/validation.py), the last ~20 bars of a test window can rely on swing/pivot features
that were confirmed using bars just past the window's own end - a known, small, already-characterized
edge effect, not a new one. Re-running the no-lookahead check itself per fold is not done here (too
expensive for a sweep and already proven per-strategy in scripts/backtest_strategies.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from backend.research.param_sweep import ParamGrid, plateau_fraction, pick_best, sweep
from backend.risk.risk_manager import RiskLimits
from backend.research.simple_backtester import BacktestResult, SimTrade, simulate_signals
from backend.strategies.base import Strategy, StrategyContext
from backend.strategies.registry import build_strategy

MIN_FOLD_BARS = 30


@dataclass
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass
class FoldResult:
    fold: Fold
    chosen_params: dict[str, Any] | None
    in_sample_summary: dict[str, Any]
    out_of_sample_trades: list[SimTrade] = field(default_factory=list)
    plateau_fraction: float = 0.0

    @property
    def out_of_sample_summary(self) -> dict[str, Any]:
        return BacktestResult(trades=self.out_of_sample_trades).summary()


@dataclass
class WalkForwardReport:
    strategy_id: str
    folds: list[FoldResult] = field(default_factory=list)

    def combined_out_of_sample_summary(self) -> dict[str, Any]:
        all_trades = [t for f in self.folds for t in f.out_of_sample_trades]
        return BacktestResult(trades=all_trades).summary()

    def walk_forward_efficiency(self) -> float | None:
        """Mean of (OOS expectancy / IS expectancy) over folds where in-sample was actually profitable -
        undefined (and misleading) otherwise, so those folds are excluded rather than counted as 0."""
        ratios = [f.out_of_sample_summary.get("expectancy", 0.0) / f.in_sample_summary["expectancy"]
                 for f in self.folds
                 if f.in_sample_summary.get("expectancy", 0) > 0 and f.out_of_sample_summary.get("trades", 0) > 0]
        return sum(ratios) / len(ratios) if ratios else None


def generate_folds(min_ts: pd.Timestamp, max_ts: pd.Timestamp, train_days: int, test_days: int,
                   step_days: int | None = None) -> list[Fold]:
    """Rolling (fixed-size, sliding) train/test windows spanning [min_ts, max_ts]. `step_days` defaults
    to `test_days`, giving non-overlapping test windows - the simplest, easiest-to-reason-about scheme."""
    step_days = step_days or test_days
    start = pd.Timestamp(min_ts).normalize()
    end = pd.Timestamp(max_ts).normalize()
    folds = []
    train_start = start
    while True:
        train_end = train_start + pd.Timedelta(days=train_days)
        test_start = train_end
        test_end = test_start + pd.Timedelta(days=test_days)
        if test_end > end:
            break
        folds.append(Fold(train_start, train_end, test_start, test_end))
        train_start = train_start + pd.Timedelta(days=step_days)
    return folds


def _to_frame_tz(bound: pd.Timestamp, tz) -> pd.Timestamp:
    """`bound` may be naive (needs localizing to `tz`) or already tz-aware in some other zone (needs
    converting) - `generate_folds` inherits whatever tz-awareness the caller's min/max timestamps had,
    which is naive for synthetic/test data but tz-aware for real (yfinance/Angel One) data."""
    if tz is None:
        return bound
    return bound.tz_localize(tz) if bound.tzinfo is None else bound.tz_convert(tz)


def _slice(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    ts = df["ts"]
    tz = ts.dt.tz
    start, end = _to_frame_tz(start, tz), _to_frame_tz(end, tz)
    mask = (ts >= start) & (ts < end)
    return df.loc[mask].reset_index(drop=True)


def run_walk_forward(cls: type[Strategy], version: str, grid: ParamGrid,
                     feature_by_symbol: dict[str, pd.DataFrame], timeframe: str, folds: list[Fold],
                     capital_per_trade: float = 200_000.0, min_trades: int = 30,
                     progress: Callable[[int, int, Fold], None] | None = None,
                     entry_fill_mode: str = "next_open", limits: RiskLimits | None = None) -> WalkForwardReport:
    report = WalkForwardReport(strategy_id=cls.strategy_id)
    for n, fold in enumerate(folds, 1):
        if progress:
            progress(n, len(folds), fold)
        train_feat = {s: sl for s, df in feature_by_symbol.items()
                     if len(sl := _slice(df, fold.train_start, fold.train_end)) >= MIN_FOLD_BARS}
        test_feat = {s: sl for s, df in feature_by_symbol.items()
                    if len(sl := _slice(df, fold.test_start, fold.test_end)) >= MIN_FOLD_BARS}

        candidates = (sweep(cls, version, grid, train_feat, timeframe, capital_per_trade, entry_fill_mode, limits)
                      if train_feat else [])
        best = pick_best(candidates, min_trades=min_trades) if candidates else None
        if best is None:
            report.folds.append(FoldResult(fold, None, {"trades": 0}))
            continue

        pf = plateau_fraction(best, candidates, grid)
        oos_trades: list[SimTrade] = []
        strat = build_strategy(cls, version, overrides=best.params)
        for symbol, feat in test_feat.items():
            signals = strat.run(StrategyContext(symbol, timeframe, feat))
            oos_trades.extend(simulate_signals(feat, signals, capital_per_trade=capital_per_trade,
                                               entry_fill_mode=entry_fill_mode, limits=limits).trades)

        report.folds.append(FoldResult(fold, best.params, best.summary, oos_trades, pf))
    return report
