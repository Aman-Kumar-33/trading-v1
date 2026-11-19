"""Build the per-signal intraday edge dataset (see backend/research/edge_diagnostics.py) for many symbols,
in parallel, using the feature cache. Writes backtest_data/results/intraday_edges/signals.parquet and
prints gross / cost / net R per strategy for the discovery and holdout periods.

    python scripts/intraday_edge_report.py --universe nifty50 --limit 20 --workers 12
"""
import argparse
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from _bootstrap import init

HOLDOUT_START = "2025-10-01"   # last ~12 months are never used to choose rules


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols")
    p.add_argument("--universe", default="nifty50")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--timeframe", default="5m")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--ticket", type=float, default=200_000.0)
    return p.parse_args()


def _one_symbol(symbol: str, timeframe: str, ticket: float) -> list[dict]:
    from backend.core.config import get_settings
    from backend.features.cache import cached_build_features
    from backend.research.edge_diagnostics import UNGATED, signal_outcomes
    from backend.storage import parquet_store
    from backend.strategies import INTRADAY_STRATEGIES
    from backend.strategies.registry import build_strategy

    settings = get_settings()

    def load(tf):
        df = parquet_store.read(settings.paths.clean / tf / f"{parquet_store.file_stem(symbol)}.parquet")
        if df is None or df.empty:
            return None
        if tf != "1d" and "day_ok" in df.columns:
            df = df.loc[df["day_ok"]]
        return df.reset_index(drop=True)

    raw, daily = load(timeframe), load("1d")
    if raw is None or len(raw) < 500:
        return []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feat = cached_build_features(settings.paths.backtest_data / "features_cache", symbol, raw, timeframe, daily)
        rows = []
        for cls in INTRADAY_STRATEGIES:
            if cls.strategy_id in {"pairs_spread_reversion", "lead_lag"}:
                continue
            rows.extend(signal_outcomes(feat, build_strategy(cls, "v1", overrides=UNGATED), symbol, timeframe, ticket))
    return rows


def main() -> int:
    args = parse_args()
    settings, log = init("intraday_edge_report")
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        from backend.data.universe import load_universe
        symbols = load_universe(settings, args.universe)[: args.limit]
    print(f"  {len(symbols)} symbols, {args.workers} workers")
    t0 = time.time()
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_one_symbol, s, args.timeframe, args.ticket): s for s in symbols}
        for n, fut in enumerate(as_completed(futures), 1):
            sym = futures[fut]
            try:
                part = fut.result()
                rows.extend(part)
                print(f"    [{n}/{len(symbols)}] {sym}: {len(part)} signals ({time.time() - t0:.0f}s)", flush=True)
            except Exception as exc:  # noqa: BLE001
                log.exception("symbol failed", extra={"data": {"symbol": sym, "error": repr(exc)}})
                print(f"    [{n}/{len(symbols)}] {sym}: FAILED {exc!r}", flush=True)
    df = pd.DataFrame(rows)
    if df.empty:
        print("  no signals")
        return 1
    df["period"] = (pd.to_datetime(df["ts"]).dt.tz_localize(None) >= pd.Timestamp(HOLDOUT_START)).map(
        {False: "discovery", True: "holdout"})
    out = settings.paths.results / "intraday_edges"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "signals.parquet")
    from backend.research.edge_diagnostics import summarize
    pd.set_option("display.width", 200)
    print(summarize(df, ["strategy", "period"]).to_string())
    print(f"\n  {len(df)} signals in {time.time() - t0:.0f}s -> {out / 'signals.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
