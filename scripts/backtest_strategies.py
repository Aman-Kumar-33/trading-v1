"""Backtest every strategy in the pool against real historical data, fed through the exact same
build_features -> strategy.run() path live trading uses - so strategies can't cheat, not by trusting
that alone but by proving it (see backend/research/validation.py's no-lookahead check, run for every
symbol/strategy pair here, not just in the test suite).

Features are built once per symbol/timeframe and shared across every strategy that uses that timeframe
(they don't depend on which strategy reads them) - rebuilding per strategy would be ~13x more expensive
for no benefit. Same for the no-lookahead check's per-cutoff feature sets.

Pairs strategies (pairs_spread_reversion, pairs_multiday) and lead_lag need a second instrument's data
wired up per-pair/per-leader (see tests/test_integration_phase2.py for the pattern) - skipped here, not
yet part of this first backtest pass.

Examples:
    python scripts/backtest_strategies.py --symbols RELIANCE,TCS,INFY               # quick smoke test
    python scripts/backtest_strategies.py --universe nifty50 --long-term-only       # bigger, daily only
    python scripts/backtest_strategies.py --universe nifty100                       # the full first pass

Designed to run for a while over a big universe - that's fine, it's not meant to be watched turn by
turn. Results land in backtest_data/results/<strategy_id>/<version>/, ready for
scripts/backtest_report.py. Every symbol x strategy run is logged to logs/backtest_strategies.jsonl with
the full exception on failure, so a bad run is diagnosable without re-running it.
"""
import argparse
import json
import time
import warnings
from collections import defaultdict

import pandas as pd

from _bootstrap import init

from backend.features.pipeline import build_features
from backend.research.simple_backtester import BacktestResult, simulate_signals
from backend.research.validation import build_cutoff_features, check_no_lookahead_with_features
from backend.risk.risk_manager import RiskLimits
from backend.storage import parquet_store
from backend.strategies import INTRADAY_STRATEGIES, LONGTERM_STRATEGIES
from backend.strategies.base import StrategyContext
from backend.strategies.registry import build_strategy

NEEDS_EXTRA_DATA = {"pairs_spread_reversion", "pairs_multiday", "lead_lag"}
# Per-trade rules live applies (no delivery shorts, fee hurdles) - capital isn't used by these checks.
LIVE_TRADE_RULES = RiskLimits(capital=1.0)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", help="comma separated symbols (overrides --universe)")
    p.add_argument("--universe", default="nifty50")
    p.add_argument("--limit", type=int, default=10, help="cap the number of symbols (default: 10, a quick pass)")
    p.add_argument("--intraday-timeframe", default="5m")
    p.add_argument("--intraday-bars", type=int, default=4000, help="how many recent intraday bars to load")
    p.add_argument("--daily-bars", type=int, default=1500, help="how many recent daily bars to load")
    p.add_argument("--capital-per-trade", type=float, default=200_000.0,
                   help="fixed ticket per trade; Rs 20,000 (the old default) makes Rs 20 brokerage 0.1%%/leg")
    p.add_argument("--entry-fill", choices=["next_open", "signal"], default="next_open",
                   help="'next_open' matches live; 'signal' is the old, optimistic same-bar fill")
    p.add_argument("--intraday-only", action="store_true")
    p.add_argument("--long-term-only", action="store_true")
    p.add_argument("--skip-validation", action="store_true", help="skip the no-lookahead check (faster, less safe)")
    p.add_argument("--version", default="v1")
    return p.parse_args()


def pick_symbols(settings, args) -> list[str]:
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        from backend.data.universe import load_universe
        symbols = load_universe(settings, args.universe)
    return symbols[: args.limit] if args.limit else symbols


def load_clean(settings, symbol: str, timeframe: str, n_bars: int) -> pd.DataFrame | None:
    path = settings.paths.clean / timeframe / f"{parquet_store.file_stem(symbol)}.parquet"
    df = parquet_store.read(path)
    if df is None or df.empty:
        return None
    if timeframe != "1d" and "day_ok" in df.columns:
        df = df.loc[df["day_ok"]].reset_index(drop=True)
    return df.tail(n_bars).reset_index(drop=True)


def main() -> int:
    args = parse_args()
    settings, log = init("backtest_strategies")
    symbols = pick_symbols(settings, args)
    strategies = [] if args.long_term_only else list(INTRADAY_STRATEGIES)
    strategies += [] if args.intraday_only else list(LONGTERM_STRATEGIES)
    strategies = [c for c in strategies if c.strategy_id not in NEEDS_EXTRA_DATA]
    strategy_by_timeframe: dict[str, list] = defaultdict(list)
    for cls in strategies:
        tf = args.intraday_timeframe if cls in INTRADAY_STRATEGIES else "1d"
        strategy_by_timeframe[tf].append(build_strategy(cls, args.version))
    print(f"  {len(symbols)} symbols x {len(strategies)} strategies "
         f"({', '.join(c.strategy_id for c in strategies)})")

    results_root = settings.paths.results
    all_trades: dict[str, list] = defaultdict(list)      # strategy_id -> [SimTrade, ...]
    lookahead_failures: dict[str, list] = defaultdict(list)
    symbols_seen: dict[str, set[str]] = defaultdict(set)
    symbols_run: dict[str, set[str]] = defaultdict(set)
    t0 = time.time()

    for timeframe, strats in strategy_by_timeframe.items():
        n_bars = args.intraday_bars if timeframe != "1d" else args.daily_bars
        for symbol in symbols:
            raw = load_clean(settings, symbol, timeframe, n_bars)
            if raw is None or len(raw) < 60:
                log.warning("no usable clean data - skipped", extra={"data": {"symbol": symbol, "timeframe": timeframe}})
                continue
            daily = load_clean(settings, symbol, "1d", 100_000) if timeframe != "1d" else None
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    feat = build_features(raw, timeframe, daily=daily)   # built ONCE for this symbol/timeframe
                    cutoff_feat = None if args.skip_validation else build_cutoff_features(raw, timeframe, daily=daily)
            except Exception as exc:  # noqa: BLE001
                log.exception("feature build failed - skipping this symbol for all strategies of this timeframe",
                             extra={"data": {"symbol": symbol, "timeframe": timeframe, "error": repr(exc)}})
                continue

            for strat in strats:
                symbols_seen[strat.strategy_id].add(symbol)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error")
                        signals = strat.run(StrategyContext(symbol, timeframe, feat))
                        if cutoff_feat is not None:
                            report = check_no_lookahead_with_features(strat, cutoff_feat, symbol, timeframe)
                            if not report.passed:
                                lookahead_failures[strat.strategy_id].append(
                                    {"symbol": symbol, "mismatches": report.mismatches[:5]})
                                log.error("NO-LOOKAHEAD CHECK FAILED", extra={"data": {
                                    "strategy": strat.strategy_id, "symbol": symbol, "mismatches": report.mismatches[:5]}})
                                continue   # don't trust results for a strategy/symbol that failed the check
                        result = simulate_signals(feat, signals, capital_per_trade=args.capital_per_trade,
                                                  entry_fill_mode=args.entry_fill, limits=LIVE_TRADE_RULES)
                    symbols_run[strat.strategy_id].add(symbol)
                    all_trades[strat.strategy_id].extend(result.trades)
                    log.info("backtested", extra={"data": {"strategy": strat.strategy_id, "symbol": symbol,
                                                           "bars": len(raw), "signals": len(signals),
                                                           "trades": len(result.trades)}})
                except Exception as exc:  # noqa: BLE001 - one bad symbol/strategy must not stop the whole run
                    log.exception("backtest failed for this symbol", extra={"data": {
                        "strategy": strat.strategy_id, "symbol": symbol, "timeframe": timeframe, "error": repr(exc)}})

    overall = {"symbols": symbols, "strategies": [c.strategy_id for c in strategies], "per_strategy": {}}
    for cls in strategies:
        sid = cls.strategy_id
        trades = all_trades[sid]
        out_dir = results_root / sid / args.version
        out_dir.mkdir(parents=True, exist_ok=True)
        if trades:
            parquet_store.write_atomic(pd.DataFrame([t.to_dict() for t in trades]), out_dir / "trades.parquet")
        summary = BacktestResult(trades=trades).summary()
        summary.update({"symbols_with_data": len(symbols_seen[sid]), "symbols_run": len(symbols_run[sid]),
                        "lookahead_failures": len(lookahead_failures[sid]),
                        "timeframe": args.intraday_timeframe if cls in INTRADAY_STRATEGIES else "1d"})
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        if lookahead_failures[sid]:
            (out_dir / "lookahead_failures.json").write_text(
                json.dumps(lookahead_failures[sid], indent=2, default=str), encoding="utf-8")
        overall["per_strategy"][sid] = summary
        print(f"  {sid:28s} {summary.get('trades', 0):5d} trades  win_rate={summary.get('win_rate', 0):.0%}  "
             f"net_pnl={summary.get('net_pnl', 0):>10.0f}  "
             f"({len(symbols_run[sid])}/{len(symbols_seen[sid])} symbols, "
             f"{len(lookahead_failures[sid])} lookahead failures)")

    (results_root / "backtest_run_summary.json").write_text(json.dumps(overall, indent=2, default=str), encoding="utf-8")
    log.info("backtest run complete", extra={"data": {"duration_s": round(time.time() - t0, 1),
                                                      "strategies": len(strategies), "symbols": len(symbols)}})
    print(f"\n  done in {time.time() - t0:.0f}s. Results in {results_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
