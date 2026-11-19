"""No-lookahead validation: proves a strategy's signals at time T don't depend on data after T, by
rebuilding features from raw OHLCV truncated at several points and checking that every signal produced
with less history available is *identical* to what the full-history run produced at that same
timestamp. If a signal could see the future, giving it less of the future to see would change it.

This is stronger than just replaying candles (which only proves candle-building is faithful, already
covered by tests/test_replay_pipeline.py's parity check) - it exercises the real
raw -> build_features -> strategy.run() path end to end, the same path backtest_strategies.py uses for
its main pass, which is exactly what needs proving.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from backend.core.models import Signal
from backend.features.pipeline import build_features
from backend.strategies.base import Strategy, StrategyContext

DEFAULT_CUTOFF_FRACTIONS = (0.5, 0.75, 0.9, 1.0)
MIN_BARS_FOR_WARMUP = 260   # enough for the slowest indicator (EMA/rolling-percentile ~252) to have real values

# Swing-pivot-based features (backend.features.price_action / patterns.chart_patterns) need `right`
# future bars to *confirm* a pivot, and pattern breakout search looks up to `max_breakout_bars` ahead.
# So a signal in the very last handful of bars of a truncated cutoff is inherently provisional *for that
# cutoff's own snapshot* - a live system has the exact same characteristic (today's newest pattern read
# can revise once tomorrow's bars confirm nearby swings). This isn't lookahead - a signal that changes
# here never saw data past its own cutoff - so these edge bars are excluded from the mismatch check
# (still counted in signal_counts). Default matches chart_patterns.py's typical breakout search window.
EDGE_BUFFER_BARS = 20


@dataclass
class LookaheadReport:
    symbol: str
    strategy_id: str
    cutoff_fractions: tuple[float, ...]
    signal_counts: dict[float, int] = field(default_factory=dict)
    mismatches: list[dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.mismatches


def _signal_key(sig: Signal) -> tuple:
    return (sig.side.value, round(sig.entry, 6) if sig.entry is not None else None,
           round(sig.stop_loss, 6), tuple(round(t, 6) for t in sig.targets))


def build_cutoff_features(raw_df: pd.DataFrame, timeframe: str,
                          cutoff_fractions: tuple[float, ...] = DEFAULT_CUTOFF_FRACTIONS,
                          daily: pd.DataFrame | None = None) -> dict[float, pd.DataFrame]:
    """The expensive part (rebuilding features from raw at each truncation) - shared across every
    strategy for this symbol/timeframe, since features don't depend on which strategy will read them.
    Call this once per symbol, then check_no_lookahead_with_features() once per strategy. `daily` isn't
    truncated: the daily_trend column for a bar on day D only ever reads days before D, so a longer daily
    history can't leak anything - and the check would catch it if it did."""
    out: dict[float, pd.DataFrame] = {}
    for frac in sorted(cutoff_fractions):
        n = max(MIN_BARS_FOR_WARMUP, int(len(raw_df) * frac))
        n = min(n, len(raw_df))
        out[frac] = build_features(raw_df.iloc[:n].reset_index(drop=True), timeframe, daily=daily)
    return out


def check_no_lookahead_with_features(strategy: Strategy, features_by_cutoff: dict[float, pd.DataFrame],
                                     symbol: str, timeframe: str, edge_buffer_bars: int = EDGE_BUFFER_BARS
                                     ) -> LookaheadReport:
    """The cheap part: run one strategy against already-built per-cutoff feature sets
    (build_cutoff_features). Split out so a research run can build features once per symbol and reuse
    them across all 18 strategies, instead of rebuilding per strategy."""
    cutoff_fractions = tuple(sorted(features_by_cutoff))
    report = LookaheadReport(symbol=symbol, strategy_id=strategy.strategy_id, cutoff_fractions=cutoff_fractions)
    if not features_by_cutoff:
        return report   # too little data for a meaningful check either way - not a failure, just skipped

    reference_frac = max(cutoff_fractions)
    signals_by_cutoff: dict[float, dict[pd.Timestamp, Signal]] = {}
    edge_ts_by_cutoff: dict[float, pd.Timestamp | None] = {}
    for frac, feat in features_by_cutoff.items():
        sigs = strategy.run(StrategyContext(symbol, timeframe, feat))
        signals_by_cutoff[frac] = {s.ts: s for s in sigs}
        report.signal_counts[frac] = len(sigs)
        edge_ts_by_cutoff[frac] = feat["ts"].iloc[-edge_buffer_bars] if len(feat) > edge_buffer_bars else None

    reference = signals_by_cutoff[reference_frac]
    for frac, sigs in signals_by_cutoff.items():
        if frac == reference_frac:
            continue
        edge_ts = edge_ts_by_cutoff[frac]
        for ts, sig in sigs.items():
            if edge_ts is not None and ts >= edge_ts:
                continue   # within edge_buffer_bars of this cutoff's own end - provisional, not a violation
            ref = reference.get(ts)
            if ref is None or _signal_key(ref) != _signal_key(sig):
                report.mismatches.append({
                    "cutoff_frac": frac, "ts": str(ts),
                    "truncated": {"side": sig.side.value, "entry": sig.entry, "stop": sig.stop_loss, "targets": sig.targets},
                    "reference": None if ref is None else {"side": ref.side.value, "entry": ref.entry,
                                                            "stop": ref.stop_loss, "targets": ref.targets},
                })
    return report


def check_no_lookahead(strategy: Strategy, raw_df: pd.DataFrame, symbol: str, timeframe: str,
                       cutoff_fractions: tuple[float, ...] = DEFAULT_CUTOFF_FRACTIONS) -> LookaheadReport:
    """Convenience one-shot version (builds features itself) - for checking a single strategy/symbol in
    isolation, e.g. in tests. A research run checking many strategies against the same symbol should call
    build_cutoff_features() once and check_no_lookahead_with_features() per strategy instead."""
    if len(raw_df) < MIN_BARS_FOR_WARMUP:
        return LookaheadReport(symbol=symbol, strategy_id=strategy.strategy_id, cutoff_fractions=cutoff_fractions)
    features_by_cutoff = build_cutoff_features(raw_df, timeframe, cutoff_fractions)
    return check_no_lookahead_with_features(strategy, features_by_cutoff, symbol, timeframe)
