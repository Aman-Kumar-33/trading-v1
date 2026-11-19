"""Backtest data for the dashboard's Backtesting page: a run picker (`list_backtest_runs`) plus a full
per-run detail (`backtest_run_detail`) with alpha/beta/sharpe and an equity curve - neither is persisted
anywhere on disk by the research scripts, so both are derived here from `trades.parquet` on request,
reusing `backend.analytics.metrics.compute_metrics` and `backend.analytics.dashboard_data`'s existing
loaders rather than re-deriving them.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from backend.analytics import dashboard_data as dd
from backend.analytics.metrics import _daily_net_pnl, compute_metrics
from backend.core.config import Settings
from backend.storage import parquet_store


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _equity_series(trades: list, capital: float, benchmark_returns: pd.Series | None,
                   beta: float | None = None) -> list[dict]:
    """One row per trading day: cumulative equity, the benchmark indexed to the same starting capital, and
    (when beta is known) cumulative alpha - the rupee P&L left over after stripping out what beta's worth
    of benchmark exposure alone would have produced. That residual, accumulating over time, *is* what
    "alpha" means - a single scalar annualized alpha (metrics.py's compute_metrics) doesn't show the
    dashboard's "graph of alpha" ask, this does."""
    daily = _daily_net_pnl(trades)
    if daily.empty:
        return []
    equity = capital + daily.cumsum()
    bench_equity = None
    alpha_cum = None
    if benchmark_returns is not None and not benchmark_returns.empty:
        aligned = benchmark_returns.reindex(equity.index).fillna(0.0)
        bench_equity = capital * (1 + aligned).cumprod()
        if beta is not None:
            daily_returns = daily / capital if capital > 0 else daily * 0.0
            alpha_daily = daily_returns - beta * aligned
            alpha_cum = (alpha_daily * capital).cumsum()
    rows = []
    for i, (ts, eq) in enumerate(equity.items()):
        row = {"ts": ts.isoformat(), "equity": round(float(eq), 2)}
        if bench_equity is not None:
            row["benchmark_equity"] = round(float(bench_equity.iloc[i]), 2)
        if alpha_cum is not None:
            row["alpha_equity"] = round(float(alpha_cum.iloc[i]), 2)
        rows.append(row)
    return rows


def _monthly_pnl(trades: list) -> list[dict]:
    daily = _daily_net_pnl(trades)
    if daily.empty:
        return []
    g = daily.groupby([daily.index.year, daily.index.month]).sum()
    return [{"year": int(y), "month": int(m), "net_pnl": round(float(v), 2)} for (y, m), v in g.items()]


def _run_detail_from_trades(d: Path, kind: str, run_id: str, settings: Settings) -> dict:
    summary = _read_json(d / "summary.json")
    trades_path = d / "trades.parquet"
    trades_df = parquet_store.read(trades_path) if trades_path.exists() else None
    trades = dd._trades_from_df(trades_df)
    capital = dd.load_capital(settings)["paper_capital"]
    bench = dd.benchmark_returns(settings)
    metrics = compute_metrics(trades, capital, bench).as_dict()
    return {"kind": kind, "id": run_id, "summary": summary, "metrics": metrics,
           "equity_curve": _equity_series(trades, capital, bench, metrics["beta"]), "monthly_pnl": _monthly_pnl(trades)}


def live_strategy_equity(settings: Settings, conn, strategy_id: str, mode: str) -> list[dict]:
    """Paper/live equity curve for one strategy, straight from the trades DB - same shape as a backtest
    run's equity_curve, so the Strategies page inspector can plot "live strategy performance" with the
    same chart component, not a bespoke one."""
    rows = dd.db_trades(conn, mode, strategy_id)
    capital = dd.load_capital(settings)["paper_capital" if mode == "paper" else "capital"] or \
        dd.load_capital(settings)["paper_capital"]
    return _equity_series(rows, capital, None)


def list_backtest_runs(settings: Settings) -> list[dict]:
    root = settings.paths.results
    out = []
    for summary_path in sorted(root.glob("*/v1/summary.json")):
        sid = summary_path.parent.parent.name
        s = _read_json(summary_path)
        out.append({"kind": "strategy", "id": sid, "label": sid, "trades": s.get("trades", 0),
                   "net_pnl": s.get("net_pnl")})

    fs_root = root / "full_system"
    bare_summary = fs_root / "summary.json"
    if bare_summary.exists():
        s = _read_json(bare_summary)
        overall = s.get("overall", s)
        out.append({"kind": "full_system", "id": "default", "label": "full_system (default)",
                   "trades": overall.get("trades", 0), "net_pnl": overall.get("net_pnl")})
    if fs_root.exists():
        for run_dir in sorted(p for p in fs_root.iterdir() if p.is_dir()):
            sp = run_dir / "summary.json"
            if not sp.exists():
                continue
            s = _read_json(sp)
            overall = s.get("overall", s)
            out.append({"kind": "full_system", "id": run_dir.name, "label": f"full_system/{run_dir.name}",
                       "trades": overall.get("trades", 0), "net_pnl": overall.get("net_pnl")})

    for summary_path in sorted(root.glob("walk_forward/*/v1/summary.json")):
        sid = summary_path.parent.parent.name
        s = _read_json(summary_path)
        coos = s.get("combined_out_of_sample", {})
        out.append({"kind": "walk_forward", "id": sid, "label": f"walk-forward/{sid}",
                   "trades": coos.get("trades", 0), "net_pnl": coos.get("net_pnl"),
                   "walk_forward_efficiency": s.get("walk_forward_efficiency")})
    return out


def backtest_run_detail(settings: Settings, kind: str, run_id: str) -> dict:
    root = settings.paths.results
    if kind == "strategy":
        d = root / run_id / "v1"
        if not (d / "summary.json").exists():
            raise FileNotFoundError(f"no strategy backtest result for {run_id!r}")
        return _run_detail_from_trades(d, kind, run_id, settings)
    if kind == "full_system":
        d = root / "full_system" if run_id == "default" else root / "full_system" / run_id
        if not (d / "summary.json").exists():
            raise FileNotFoundError(f"no full-system backtest result for {run_id!r}")
        return _run_detail_from_trades(d, kind, run_id, settings)
    if kind == "walk_forward":
        d = root / "walk_forward" / run_id / "v1"
        if not (d / "summary.json").exists():
            raise FileNotFoundError(f"no walk-forward result for {run_id!r}")
        folds = _read_json(d / "folds.json") if (d / "folds.json").exists() else []
        summary = _read_json(d / "summary.json")
        return {"kind": kind, "id": run_id, "summary": summary, "metrics": None, "equity_curve": None,
               "monthly_pnl": None, "folds": folds}
    raise ValueError(f"unknown backtest kind: {kind!r}")
