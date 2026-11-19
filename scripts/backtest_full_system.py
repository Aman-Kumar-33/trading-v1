"""Backtest the WHOLE system together - every strategy's signals flowing through the same Allocator and
RiskManager objects Phase 4 built for live/paper trading, not strategy-by-strategy in isolation
(scripts/backtest_strategies.py) and not tick-by-tick single-day replay (scripts/paper_trade_replay.py,
too slow for a multi-month run). See backend/research/full_system_backtest.py for the engine itself.

Live-parity by default: entries fill at the next bar's open, stops gap-fill at the open, every Angel One
charge (incl. DP on delivery sells) is applied, delivery shorts are rejected, size = risk % x ensemble
weight capped by notional/exposure/volume, and trades whose fees would eat >25% of their risk are
skipped. Each run gets its own results folder (nothing is overwritten) including the ensemble's learned
state (tracker.json), which live trading can warm-start from.

Pairs/lead_lag strategies are skipped (they need a second instrument's data not wired up here yet).

    python scripts/backtest_full_system.py --symbols RELIANCE,TCS,INFY,HDFCBANK,ICICIBANK   # quick check
    python scripts/backtest_full_system.py --universe nifty50 --limit 30                     # bigger
    python scripts/backtest_full_system.py --strategies promoted --universe nifty50 --limit 30   # what would really trade

`--capital` is a placeholder (default 1,000,000) - pass your real number if you want the risk-manager
sizing/limits to mean anything close to your actual situation.
"""
import argparse
import json
import time as time_module
import warnings
from collections import defaultdict
from datetime import datetime

import pandas as pd

from _bootstrap import init

from backend.ensemble.allocator import Allocator
from backend.ensemble.performance_tracker import PerformanceTracker
from backend.features.pipeline import build_features
from backend.research.full_system_backtest import run_full_system_backtest
from backend.research.simple_backtester import ENTRY_FILL_MODES
from backend.risk.risk_manager import RiskLimits, RiskManager
from backend.storage import parquet_store
from backend.storage.db import init_db
from backend.strategies import INTRADAY_STRATEGIES, LONGTERM_STRATEGIES
from backend.strategies.registry import build_strategy, get_status

NEEDS_EXTRA_DATA = {"pairs_spread_reversion", "pairs_multiday", "lead_lag"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", help="comma separated symbols (overrides --universe)")
    p.add_argument("--universe", default="nifty50")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--strategies", choices=["all", "promoted"], default="all",
                   help="'all' = every wired-up strategy (see how the whole pool interacts with the "
                        "allocator); 'promoted' = only strategy_versions.status PAPER/LIVE (what would "
                        "actually trade right now)")
    p.add_argument("--intraday-timeframe", default="5m")
    p.add_argument("--intraday-bars", type=int, default=100_000, help="cap on recent intraday bars loaded per symbol")
    p.add_argument("--daily-bars", type=int, default=100_000, help="cap on recent daily bars loaded per symbol")
    p.add_argument("--capital", type=float, default=1_000_000.0, help="PLACEHOLDER - set to your real capital")
    p.add_argument("--long-term-capital-pct", type=float, default=0.2,
                   help="share of capital for long-term (delivery) trades; the rest is intraday. The plan's "
                        "split is 80/20 intraday/long-term - this lets you test others")
    p.add_argument("--risk-pct-per-trade", type=float, default=0.01)
    p.add_argument("--max-concurrent-positions", type=int, default=5)
    p.add_argument("--daily-loss-limit-pct", type=float, default=0.03)
    p.add_argument("--intraday-leverage", type=float, default=1.0, help="MIS leverage (1 = none; SEBI allows ~5)")
    p.add_argument("--max-position-pct", type=float, default=0.25, help="max one position's share of its bucket")
    p.add_argument("--max-cost-to-risk-intraday", type=float, default=0.10, help="skip intraday trades whose fees exceed this x risk")
    p.add_argument("--max-cost-to-risk-long-term", type=float, default=0.25, help="same for delivery trades")
    p.add_argument("--min-reward-to-cost", type=float, default=3.0, help="if a target exists, reward must be >= this x fees")
    p.add_argument("--max-entries-per-symbol-per-day", type=int, default=2)
    p.add_argument("--carry-days", type=int, default=3,
                   help="intraday longs not exited by square-off are carried up to this many days (0 = square off)")
    p.add_argument("--entry-fill", choices=ENTRY_FILL_MODES, default="next_open",
                   help="'next_open' matches live; 'signal' is the old, optimistic same-bar fill")
    p.add_argument("--run-name", help="results subfolder (default: <strategies>_<timestamp>)")
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


def _fmt(s: dict) -> str:
    gross, costs, net = s.get("gross_pnl", 0), s.get("total_costs", 0), s.get("net_pnl", 0)
    return (f"{s.get('trades', 0):5d} trades  win={s.get('win_rate', 0):4.0%}  gross={gross:>11,.0f}  "
            f"fees={costs:>10,.0f}  net={net:>11,.0f}")


def main() -> int:
    args = parse_args()
    settings, log = init("backtest_full_system")
    symbols = pick_symbols(settings, args)
    conn = init_db(settings.paths.db_file)

    classes = [c for c in (*INTRADAY_STRATEGIES, *LONGTERM_STRATEGIES) if c.strategy_id not in NEEDS_EXTRA_DATA]
    if args.strategies == "promoted":
        classes = [c for c in classes if get_status(conn, c.strategy_id, args.version) in ("PAPER", "LIVE")]
    if not classes:
        print("  no strategies match --strategies filter - nothing to backtest")
        return 1
    strategies_by_timeframe: dict[str, list] = defaultdict(list)
    for cls in classes:
        tf = args.intraday_timeframe if cls in INTRADAY_STRATEGIES else "1d"
        strategies_by_timeframe[tf].append(build_strategy(cls, args.version))
    print(f"  {len(symbols)} symbols x {len(classes)} strategies ({args.strategies}, fill={args.entry_fill}): "
         f"{', '.join(c.strategy_id for c in classes)}")

    t0 = time_module.time()
    feature_by_symbol_tf: dict[tuple[str, str], pd.DataFrame] = {}
    for timeframe in strategies_by_timeframe:
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
                    feature_by_symbol_tf[(symbol, timeframe)] = build_features(raw, timeframe, daily=daily)
            except Exception as exc:  # noqa: BLE001
                log.exception("feature build failed - skipped", extra={"data": {
                    "symbol": symbol, "timeframe": timeframe, "error": repr(exc)}})
    if not feature_by_symbol_tf:
        print("  no usable data for any symbol - aborting")
        return 1

    risk_manager = RiskManager(RiskLimits(
        capital=args.capital, long_term_capital_pct=args.long_term_capital_pct,
        intraday_capital_pct=1.0 - args.long_term_capital_pct, risk_pct_per_trade=args.risk_pct_per_trade,
        max_concurrent_positions=args.max_concurrent_positions, daily_loss_limit_pct=args.daily_loss_limit_pct,
        intraday_leverage=args.intraday_leverage, max_position_pct=args.max_position_pct,
        max_cost_to_risk_intraday=args.max_cost_to_risk_intraday,
        max_cost_to_risk_long_term=args.max_cost_to_risk_long_term, min_reward_to_cost=args.min_reward_to_cost,
        max_entries_per_symbol_per_day=args.max_entries_per_symbol_per_day))
    tracker = PerformanceTracker()
    allocator = Allocator(tracker=tracker)

    from backend.data.universe import load_universe
    nifty50_symbols = set(load_universe(settings, "nifty50"))
    result = run_full_system_backtest(feature_by_symbol_tf, strategies_by_timeframe, risk_manager,
                                      allocator, tracker, entry_fill_mode=args.entry_fill, carry_days=args.carry_days,
                                      nifty50_symbols=nifty50_symbols)

    overall = result.summary()
    per_strategy = result.per_strategy_summary()
    rejections = dict(sorted(result.rejections.items()))

    run_name = args.run_name or f"{args.strategies}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = settings.paths.results / "full_system" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if result.trades:
        parquet_store.write_atomic(pd.DataFrame([t.to_dict() for t in result.trades]), out_dir / "trades.parquet")
    tracker.save(out_dir / "tracker.json")
    (out_dir / "summary.json").write_text(json.dumps({
        "args": vars(args), "overall": overall, "per_strategy": per_strategy, "rejections": rejections,
        "allocator_merges": result.allocator_merges, "shadow_samples": result.shadow_samples,
        "ensemble_weights": tracker.stats(), "symbols": symbols, "strategies": [c.strategy_id for c in classes],
    }, indent=2, default=str), encoding="utf-8")

    print(f"\n  overall:  {_fmt(overall)}")
    if overall.get("gross_pnl"):
        print(f"            fees = {overall['total_costs'] / abs(overall['gross_pnl']):.0%} of |gross|")
    print("\n  per strategy:")
    for sid, s in sorted(per_strategy.items(), key=lambda kv: -kv[1].get("net_pnl", 0)):
        print(f"    {sid:26s} {_fmt(s)}")
    print(f"\n  not taken: {rejections}")
    print(f"  ensemble: {result.allocator_merges} merged (confluence) entries, "
          f"{result.shadow_samples} shadow outcomes learned")
    log.info("full system backtest complete", extra={"data": {"duration_s": round(time_module.time() - t0, 1),
                                                               "overall": overall, "rejections": rejections,
                                                               "run": run_name}})
    print(f"\n  done in {time_module.time() - t0:.0f}s. Results in {out_dir}")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
