"""Backtests every intraday strategy - including the cross-sectional ones (pairs_spread_reversion,
lead_lag) that need a two-symbol context `backtest_strategies.py` can't provide - then reports one
combined, whole-system total. Reuses `backtest_strategies.py` unchanged (as a subprocess) for every
single-symbol strategy rather than duplicating it; only the cross-sectional piece and the final combined
total are new.

    python scripts/backtest_all_intraday.py --universe nifty50 --limit 40
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from _bootstrap import init

from backend.data.universe import load_universe
from backend.features import pairs as pr
from backend.features.pipeline import build_features
from backend.research.simple_backtester import BacktestResult, simulate_signals
from backend.risk.risk_manager import RiskLimits
from backend.storage import parquet_store
from backend.strategies.base import StrategyContext
from backend.strategies.intraday.lead_lag import LeadLag
from backend.strategies.intraday.pairs_spread_reversion import PairsSpreadReversion

LIVE_TRADE_RULES = RiskLimits(capital=1.0)   # same cost-hurdle ratios live/paper applies (see backtest_strategies.py)
MAX_PAIRS_TESTED = 15    # top-p-value pairs only - keeps the O(n^2) cointegration scan's downstream cost bounded


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--universe", default="nifty50")
    p.add_argument("--limit", type=int, default=40)
    p.add_argument("--timeframe", default="5m")
    p.add_argument("--bars", type=int, default=4000)
    p.add_argument("--capital-per-trade", type=float, default=200_000.0)
    p.add_argument("--version", default="v1")
    p.add_argument("--skip-validation", action="store_true",
                   help="passed through to backtest_strategies.py - faster, less safe (see its own --help)")
    return p.parse_args()


def _load(settings, symbol: str, timeframe: str, n: int) -> pd.DataFrame | None:
    df = parquet_store.read(settings.paths.clean / timeframe / f"{symbol}.parquet")
    if df is None or len(df) < 500:
        return None
    return df.tail(n).reset_index(drop=True)


def _write_result(settings, strategy_id: str, version: str, trades: list, extra: dict) -> dict:
    out_dir = settings.paths.results / strategy_id / version
    out_dir.mkdir(parents=True, exist_ok=True)
    if trades:
        parquet_store.write_atomic(pd.DataFrame([t.to_dict() for t in trades]), out_dir / "trades.parquet")
    summary = BacktestResult(trades=trades).summary()
    summary.update(extra)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def backtest_pairs(settings, feats: dict, closes: dict, timeframe: str, capital_per_trade: float, version: str) -> dict:
    price_series = {s: d.set_index("ts")["close"] for s, d in closes.items()}
    scan = pr.scan_pairs(price_series, significance=0.05, min_half_life=2, max_half_life=120)
    print(f"  {len(scan)} cointegrated pairs found among {len(closes)} symbols")
    trades = []
    for _, row in scan.head(MAX_PAIRS_TESTED).iterrows():
        y, x = row["symbol_y"], row["symbol_x"]
        strat = PairsSpreadReversion(params=PairsSpreadReversion.default_params, version=version)
        signals = strat.run(StrategyContext(y, timeframe, feats[y], extra={
            "partner_symbol": x, "partner_df": feats[x], "hedge_ratio": row["hedge_ratio"], "intercept": row["intercept"]}))
        y_price, x_price = closes[y]["close"].iloc[-1], closes[x]["close"].iloc[-1]
        # hedge-matched ticket size: X's notional tracks Y's via the hedge ratio and relative price, not
        # an independent flat allocation (two unrelated directional bets would defeat the point of a pair)
        x_capital = capital_per_trade * abs(row["hedge_ratio"]) * x_price / y_price if y_price else capital_per_trade
        ry = simulate_signals(feats[y], [s for s in signals if s.symbol == y], capital_per_trade=capital_per_trade,
                              entry_fill_mode="next_open", limits=LIVE_TRADE_RULES)
        rx = simulate_signals(feats[x], [s for s in signals if s.symbol == x], capital_per_trade=x_capital,
                              entry_fill_mode="next_open", limits=LIVE_TRADE_RULES)
        trades += ry.trades + rx.trades
    return _write_result(settings, "pairs_spread_reversion", version, trades,
                         {"pairs_tested": min(len(scan), MAX_PAIRS_TESTED), "pairs_found": len(scan)})


def backtest_lead_lag(settings, feats: dict, timeframe: str, capital_per_trade: float, version: str) -> dict:
    leader_df = _load(settings, "IDX_NSEI", timeframe, max(len(f) for f in feats.values()) if feats else 0)
    if leader_df is None:
        print("  no IDX_NSEI history - skipping lead_lag"); return {"trades": 0}
    leader_feat = build_features(leader_df, timeframe, core_only=True)
    trades = []
    for symbol, feat in feats.items():
        aligned = pd.concat([leader_feat.set_index("ts")["close"], feat.set_index("ts")["close"]],
                           axis=1, join="inner", keys=["leader", "follower"]).dropna()
        if len(aligned) < 60:
            continue
        lag, corr = pr.best_lead_lag(aligned["leader"], aligned["follower"])
        if lag <= 0 or abs(corr) < 0.2:   # follower must genuinely lag (not lead) the index, non-trivially
            continue
        strat = LeadLag(params=LeadLag.default_params, version=version)
        signals = strat.run(StrategyContext(symbol, timeframe, feat,
                                            extra={"leader_symbol": "IDX_NSEI", "leader_df": leader_feat, "lag": lag}))
        trades += simulate_signals(feat, signals, capital_per_trade=capital_per_trade,
                                   entry_fill_mode="next_open", limits=LIVE_TRADE_RULES).trades
    return _write_result(settings, "lead_lag", version, trades, {})


def main() -> int:
    args = parse_args()
    settings, log = init("backtest_all_intraday")
    symbols = load_universe(settings, args.universe)[: args.limit]

    print("== single-symbol intraday strategies (backtest_strategies.py, unchanged) ==")
    cmd = [sys.executable, str(Path(__file__).parent / "backtest_strategies.py"),
          "--universe", args.universe, "--limit", str(args.limit), "--intraday-only",
          "--intraday-timeframe", args.timeframe, "--version", args.version]
    if args.skip_validation:
        cmd.append("--skip-validation")
    subprocess.run(cmd, check=False)

    print("\n== cross-sectional strategies (new: pairs + lead-lag) ==")
    closes = {s: d for s in symbols if (d := _load(settings, s, args.timeframe, args.bars)) is not None}
    feats = {s: build_features(d, args.timeframe, core_only=True) for s, d in closes.items()}
    pairs_summary = backtest_pairs(settings, feats, closes, args.timeframe, args.capital_per_trade, args.version)
    print(f"  pairs_spread_reversion  {pairs_summary.get('trades', 0):5d} trades  "
         f"win_rate={pairs_summary.get('win_rate', 0):.0%}  net_pnl={pairs_summary.get('net_pnl', 0):>10.0f}")
    leadlag_summary = backtest_lead_lag(settings, feats, args.timeframe, args.capital_per_trade, args.version)
    print(f"  lead_lag                {leadlag_summary.get('trades', 0):5d} trades  "
         f"win_rate={leadlag_summary.get('win_rate', 0):.0%}  net_pnl={leadlag_summary.get('net_pnl', 0):>10.0f}")

    overall_path = settings.paths.results / "backtest_run_summary.json"
    overall = json.loads(overall_path.read_text(encoding="utf-8")) if overall_path.exists() else {"per_strategy": {}}
    overall["per_strategy"]["pairs_spread_reversion"] = pairs_summary
    overall["per_strategy"]["lead_lag"] = leadlag_summary
    total_net = sum(s.get("net_pnl", 0) for s in overall["per_strategy"].values())
    total_trades = sum(s.get("trades", 0) for s in overall["per_strategy"].values())
    print("\n== WHOLE SYSTEM COMBINED (every intraday strategy, incl. cross-sectional) ==")
    print(f"  trades={total_trades}  net_pnl={total_net:.0f}")
    overall_path.write_text(json.dumps(overall, indent=2, default=str), encoding="utf-8")
    log.info("backtest_all_intraday complete", extra={"data": {"total_trades": total_trades, "total_net_pnl": total_net}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
