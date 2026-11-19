"""Generate a self-contained HTML backtest report (equity curves, win rates, pattern performance) from
whatever scripts/backtest_strategies.py has already written to backtest_data/results/.

    python scripts/backtest_report.py                          # every strategy with results
    python scripts/backtest_report.py --strategies orb,rsi_divergence
    python scripts/backtest_report.py --out my_report.html
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from _bootstrap import init

from backend.research.report import build_html_report
from backend.storage import parquet_store
from backend.strategies import ALL_STRATEGIES


def load_benchmark(settings, symbol: str = "IDX_NSEI") -> pd.DataFrame | None:
    df = parquet_store.read(settings.paths.clean / "1d" / f"{symbol}.parquet")
    if df is None or df.empty:
        return None
    df = df.tail(1500).reset_index(drop=True)
    qty = 1
    df["indexed_pnl"] = (df["close"] - df["close"].iloc[0]) * qty
    return df[["ts", "indexed_pnl"]]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strategies", help="comma separated strategy ids (default: every strategy in the pool)")
    p.add_argument("--version", default="v1")
    p.add_argument("--out", default="backtest_report.html")
    p.add_argument("--no-benchmark", action="store_true")
    args = p.parse_args()
    settings, log = init("backtest_report")

    strategy_ids = ([s.strip() for s in args.strategies.split(",") if s.strip()] if args.strategies
                    else [c.strategy_id for c in ALL_STRATEGIES])
    summary_path = settings.paths.results / "backtest_run_summary.json"
    if not summary_path.exists():
        print("No backtest_run_summary.json found - run scripts/backtest_strategies.py first.")
        return 1
    run_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(f"  using run summary from {summary_path} ({len(run_summary.get('per_strategy', {}))} strategies)")

    benchmark = None if args.no_benchmark else load_benchmark(settings)
    out_arg = Path(args.out)
    out_path = out_arg if out_arg.is_absolute() else settings.paths.root / out_arg

    path = build_html_report(settings.paths.results, strategy_ids, out_path, args.version, benchmark)
    log.info("backtest report generated", extra={"data": {"path": str(path), "strategies": len(strategy_ids)}})
    print(f"  report written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
