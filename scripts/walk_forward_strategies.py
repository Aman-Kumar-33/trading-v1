"""Parameter sweep + walk-forward validation for the strategies that lost the most in the first
untuned backtest pass (2026-09-29): candlestick_at_level, chart_pattern_breakout, vwap_trend_pullback.

Unlike scripts/backtest_strategies.py (one fixed v1 config, straight backtest), this picks parameters on
a rolling *train* window via a grid sweep, then measures those exact parameters out-of-sample on the
following *test* window - only the OOS numbers say whether a parameter choice generalizes rather than
just fitting that window's noise. See backend/research/param_sweep.py and walk_forward.py.

Deliberately scoped small first: 3 strategies, a small grid each, a modest symbol count - prove the
machinery works and is worth trusting before spending tokens applying it to every strategy.

Example:
    python scripts/walk_forward_strategies.py --symbols RELIANCE,TCS,INFY   # quick check
    python scripts/walk_forward_strategies.py --universe nifty50 --limit 15   # the real first pass
"""
import argparse
import json
import time
import warnings

import pandas as pd

from _bootstrap import init

from backend.features.pipeline import build_features
from backend.research.walk_forward import generate_folds, run_walk_forward
from backend.risk.risk_manager import RiskLimits
from backend.storage import parquet_store
from backend.strategies import INTRADAY_STRATEGIES, LONGTERM_STRATEGIES

TARGET_STRATEGIES = ["candlestick_at_level", "chart_pattern_breakout", "vwap_trend_pullback"]

# Small, hand-picked grids - the parameters most likely to matter for each strategy's real edge, not
# every tunable in its config. target_r_multiples entries are whole lists (one atomic grid choice each).
GRIDS = {
    # intraday (5m) - already run 2026-09-30, none rescued (see strategy_vault notes)
    "candlestick_at_level": {"min_score": [70.0, 78.0, 85.0, 90.0], "stop_atr_buffer": [0.2, 0.3, 0.5]},
    "chart_pattern_breakout": {"stop_atr_mult": [1.0, 1.5, 2.0],
                              "target_r_multiples": [[1.0, 2.0], [1.5, 3.0], [2.0, 4.0]]},
    "vwap_trend_pullback": {"touch_atr_tolerance": [0.2, 0.3, 0.5], "stop_atr_buffer": [0.3, 0.5, 0.7]},
    # intraday (5m) - remaining
    "orb": {"min_atr_buffer_mult": [0.3, 0.5, 0.8], "volume_confirm_ratio": [1.0, 1.2, 1.5]},
    "vwap_reversion": {"entry_atr_mult": [1.5, 2.0, 2.5], "stop_atr_buffer": [0.2, 0.3, 0.5]},
    "supertrend_adx": {"adx_min": [15.0, 20.0, 25.0],
                       "target_r_multiples": [[1.0, 2.0], [1.5, 3.0], [2.0, 4.0]]},
    "price_action_structure": {"stop_atr_mult": [0.75, 1.0, 1.5],
                              "target_r_multiples": [[1.0, 2.0], [1.5, 3.0]]},
    "rsi_divergence": {"min_rsi_diff": [2.0, 3.0, 5.0], "stop_atr_buffer": [0.2, 0.3, 0.5]},
    "volume_spike_momentum": {"volume_spike_ratio": [2.0, 3.0, 4.0], "min_body_atr": [0.5, 0.8, 1.2]},
    "gap_fill_and_go": {"min_gap_atr": [0.3, 0.5, 0.8], "large_gap_atr": [1.5, 2.0, 3.0]},
    "news_catalyst_momentum": {"min_move_atr": [1.5, 2.5, 3.5], "news_score_min": [0.3, 0.4, 0.6]},
    # long-term (1d)
    "week52_high_momentum": {"min_days_since_last_signal": [10, 20, 30], "stop_atr_mult": [2.0, 3.0, 4.0]},
    "time_series_momentum": {"lookback_days": [126, 252, 378], "stop_atr_mult": [2.0, 3.0, 4.0]},
    "trend_ema_cross": {"stop_atr_mult": [2.0, 3.0, 4.0]},
    "quality_momentum_rank": {"top_pct": [0.6, 0.8, 0.9], "min_days_since_last_signal": [10, 20, 30]},
}
LONGTERM_IDS = {"week52_high_momentum", "time_series_momentum", "trend_ema_cross", "quality_momentum_rank"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", help="comma separated symbols (overrides --universe)")
    p.add_argument("--universe", default="nifty50")
    p.add_argument("--limit", type=int, default=15)
    p.add_argument("--strategies", default=",".join(TARGET_STRATEGIES))
    p.add_argument("--timeframe", default="5m")
    p.add_argument("--bars", type=int, default=100_000, help="cap on recent bars loaded per symbol")
    p.add_argument("--train-days", type=int, default=120)
    p.add_argument("--test-days", type=int, default=30)
    p.add_argument("--capital-per-trade", type=float, default=200_000.0,
                   help="fixed ticket per trade; Rs 20,000 (the old default) makes Rs 20 brokerage 0.1%%/leg")
    p.add_argument("--entry-fill", choices=["next_open", "signal"], default="next_open",
                   help="'next_open' matches live; 'signal' is the old, optimistic same-bar fill")
    p.add_argument("--min-trades", type=int, default=20, help="min in-sample trades for a combo to be trusted")
    p.add_argument("--version", default="v1")
    return p.parse_args()


def pick_symbols(settings, args) -> list[str]:
    if args.symbols:
        return [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    from backend.data.universe import load_universe
    symbols = load_universe(settings, args.universe)
    return symbols[: args.limit] if args.limit else symbols


def load_clean(settings, symbol: str, timeframe: str, n_bars: int) -> pd.DataFrame | None:
    path = settings.paths.clean / timeframe / f"{parquet_store.file_stem(symbol)}.parquet"
    df = parquet_store.read(path)
    if df is None or df.empty:
        return None
    if "day_ok" in df.columns:
        df = df.loc[df["day_ok"]].reset_index(drop=True)
    return df.tail(n_bars).reset_index(drop=True)


def main() -> int:
    args = parse_args()
    settings, log = init("walk_forward_strategies")
    symbols = pick_symbols(settings, args)
    strategy_ids = [s.strip() for s in args.strategies.split(",") if s.strip()]
    by_id = {c.strategy_id: c for c in (*INTRADAY_STRATEGIES, *LONGTERM_STRATEGIES)}
    classes = []
    for sid in strategy_ids:
        if sid not in GRIDS:
            print(f"  skipping {sid!r} - no parameter grid defined in this script")
            continue
        classes.append(by_id[sid])
    print(f"  {len(symbols)} symbols x {len(classes)} strategies, train={args.train_days}d test={args.test_days}d")

    t0 = time.time()
    feature_by_symbol: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        raw = load_clean(settings, symbol, args.timeframe, args.bars)
        if raw is None or len(raw) < 200:
            log.warning("no usable clean data - skipped", extra={"data": {"symbol": symbol}})
            continue
        daily = load_clean(settings, symbol, "1d", 100_000) if args.timeframe != "1d" else None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                feature_by_symbol[symbol] = build_features(raw, args.timeframe, daily=daily)   # built ONCE, shared below
        except Exception as exc:  # noqa: BLE001
            log.exception("feature build failed - skipped", extra={"data": {"symbol": symbol, "error": repr(exc)}})
    if not feature_by_symbol:
        print("  no usable data for any symbol - aborting")
        return 1

    min_ts = min(df["ts"].min() for df in feature_by_symbol.values())
    max_ts = max(df["ts"].max() for df in feature_by_symbol.values())
    folds = generate_folds(min_ts, max_ts, args.train_days, args.test_days)
    print(f"  {len(feature_by_symbol)} symbols with data, {len(folds)} walk-forward folds "
         f"({min_ts.date()} to {max_ts.date()})")
    if not folds:
        print("  not enough history for even one fold - widen the symbol set or shrink --train-days/--test-days")
        return 1

    results_root = settings.paths.results / "walk_forward"
    overall = {}
    for cls in classes:
        n_combos = 1
        for values in GRIDS[cls.strategy_id].values():
            n_combos *= len(values)
        print(f"  {cls.strategy_id}: {n_combos} combos x {len(folds)} folds...", flush=True)

        def _progress(n, total, fold, _sid=cls.strategy_id):
            print(f"    {_sid} fold {n}/{total} (train {fold.train_start.date()}..{fold.train_end.date()}, "
                 f"test ..{fold.test_end.date()})", flush=True)

        report = run_walk_forward(cls, args.version, GRIDS[cls.strategy_id], feature_by_symbol, args.timeframe,
                                  folds, capital_per_trade=args.capital_per_trade, min_trades=args.min_trades,
                                  progress=_progress, entry_fill_mode=args.entry_fill,
                                  limits=RiskLimits(capital=1.0))   # live per-trade rules; capital unused
        combined = report.combined_out_of_sample_summary()
        wfe = report.walk_forward_efficiency()
        out_dir = results_root / cls.strategy_id / args.version
        out_dir.mkdir(parents=True, exist_ok=True)
        fold_rows = [{"train_start": str(fr.fold.train_start), "test_start": str(fr.fold.test_start),
                     "test_end": str(fr.fold.test_end), "chosen_params": fr.chosen_params,
                     "plateau_fraction": fr.plateau_fraction, "in_sample": fr.in_sample_summary,
                     "out_of_sample": fr.out_of_sample_summary} for fr in report.folds]
        (out_dir / "folds.json").write_text(json.dumps(fold_rows, indent=2, default=str), encoding="utf-8")
        (out_dir / "summary.json").write_text(
            json.dumps({"combined_out_of_sample": combined, "walk_forward_efficiency": wfe}, indent=2, default=str),
            encoding="utf-8")
        overall[cls.strategy_id] = {"combined_out_of_sample": combined, "walk_forward_efficiency": wfe}
        wfe_str = f"{wfe:.2f}" if wfe is not None else "n/a"
        print(f"  {cls.strategy_id:24s} OOS: {combined.get('trades', 0):5d} trades  "
             f"win_rate={combined.get('win_rate', 0):.0%}  net_pnl={combined.get('net_pnl', 0):>10.0f}  "
             f"walk_forward_efficiency={wfe_str}")
        log.info("walk-forward complete", extra={"data": {"strategy": cls.strategy_id, **combined,
                                                          "walk_forward_efficiency": wfe}})

    (results_root / "run_summary.json").write_text(json.dumps(overall, indent=2, default=str), encoding="utf-8")
    print(f"\n  done in {time.time() - t0:.0f}s. Results in {results_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
