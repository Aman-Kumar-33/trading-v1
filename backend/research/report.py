"""Self-contained HTML backtest report: one file, no server, opens in any browser (project rule: no
servers to install). Dark theme + the validated categorical/status/sequential colors from the dataviz
skill's reference palette (references/palette.md), applied via a Plotly template rather than hand-built
SVG - Plotly's built-in hover/crosshair already satisfies the "always ship a hover layer" rule.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backend.research.simple_backtester import BacktestResult, SimTrade
from backend.storage import parquet_store

# dataviz reference palette (dark mode) - categorical slots in FIXED order (never cycled/re-sorted by rank)
SURFACE = "#1a1a19"
TEXT_PRIMARY = "#ffffff"
TEXT_SECONDARY = "#c3c2b7"
GRID = "#3a3a37"
CATEGORICAL = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
SEQUENTIAL_BLUE = ["#0d366b", "#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#5598e7", "#86b6ef", "#cde2fb"]
MAX_DIRECT_SERIES = 8   # beyond this, fold into facets rather than an unreadable rainbow (dataviz rule)

TEMPLATE = go.layout.Template(layout=go.Layout(
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
    font=dict(color=TEXT_SECONDARY, family="Segoe UI, Arial, sans-serif"),
    title=dict(font=dict(color=TEXT_PRIMARY, size=16)),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, color=TEXT_SECONDARY),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, color=TEXT_SECONDARY),
    legend=dict(bgcolor=SURFACE, font=dict(color=TEXT_SECONDARY)),
    colorway=CATEGORICAL,
))


def load_strategy_results(results_root: Path, strategy_ids: list[str], version: str = "v1"
                          ) -> dict[str, dict]:
    out = {}
    for sid in strategy_ids:
        d = results_root / sid / version
        summary_path = d / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        trades = parquet_store.read(d / "trades.parquet")
        out[sid] = {"summary": summary, "trades": trades}
    return out


def _equity_curve(trades: pd.DataFrame) -> pd.DataFrame:
    sim_trades = [SimTrade(**{**row, "entry_ts": pd.Timestamp(row["entry_ts"]), "exit_ts": pd.Timestamp(row["exit_ts"])})
                 for row in trades.to_dict("records")]
    return BacktestResult(trades=sim_trades).equity_curve()


def summary_table_figure(results: dict[str, dict]) -> go.Figure:
    rows = []
    for sid, r in sorted(results.items(), key=lambda kv: kv[1]["summary"].get("net_pnl", 0), reverse=True):
        s = r["summary"]
        rows.append([sid, s.get("trades", 0), f"{s.get('win_rate', 0):.0%}", f"{s.get('profit_factor', 0):.2f}",
                    f"{s.get('expectancy', 0):,.0f}", f"{s.get('net_pnl', 0):,.0f}", f"{s.get('max_drawdown', 0):,.0f}"])
    header = ["Strategy", "Trades", "Win Rate", "Profit Factor", "Expectancy", "Net P&L", "Max Drawdown"]
    cell_colors = [[SURFACE] * len(rows)] * len(header)
    pnl_col = 5
    for i, row in enumerate(rows):
        pnl = float(row[pnl_col].replace(",", ""))
        cell_colors[pnl_col][i] = STATUS["good"] if pnl > 0 else (STATUS["critical"] if pnl < 0 else SURFACE)
    fig = go.Figure(go.Table(
        header=dict(values=header, fill_color="#2a2a27", font=dict(color=TEXT_PRIMARY, size=12), align="left"),
        cells=dict(values=list(zip(*rows)) if rows else [[]] * len(header), fill_color=cell_colors,
                  font=dict(color=TEXT_PRIMARY, size=11), align="left", height=26)))
    fig.update_layout(template=TEMPLATE, title="Strategy Summary (v1, unoptimized defaults)",
                      margin=dict(l=10, r=10, t=40, b=10), height=80 + 30 * max(1, len(rows)))
    return fig


def equity_curves_figure(results: dict[str, dict], benchmark: pd.DataFrame | None = None) -> go.Figure:
    """One overlaid chart for up to MAX_DIRECT_SERIES strategies (ranked by trade count - the most
    statistically meaningful ones); beyond that, small-multiple facets, per the dataviz rule that a
    9th-plus series never becomes a generated/cycled hue."""
    with_curves = {sid: _equity_curve(r["trades"]) for sid, r in results.items()
                  if r["trades"] is not None and not r["trades"].empty}
    ranked = sorted(with_curves, key=lambda sid: len(with_curves[sid]), reverse=True)

    if len(ranked) <= MAX_DIRECT_SERIES:
        fig = go.Figure()
        for i, sid in enumerate(ranked):
            curve = with_curves[sid]
            fig.add_trace(go.Scatter(x=curve["exit_ts"], y=curve["equity"], mode="lines", name=sid,
                                     line=dict(color=CATEGORICAL[i % len(CATEGORICAL)], width=2)))
        if benchmark is not None:
            fig.add_trace(go.Scatter(x=benchmark["ts"], y=benchmark["indexed_pnl"], mode="lines",
                                     name="NIFTY buy & hold", line=dict(color=TEXT_SECONDARY, width=1, dash="dot")))
        fig.update_layout(template=TEMPLATE, title="Equity Curves by Strategy (simulated capital)",
                          xaxis_title="", yaxis_title="Cumulative net P&L (Rs)", height=480,
                          margin=dict(l=10, r=10, t=40, b=10))
        return fig

    cols = 3
    rows_n = -(-len(ranked) // cols)
    fig = make_subplots(rows=rows_n, cols=cols, subplot_titles=ranked)
    for i, sid in enumerate(ranked):
        curve = with_curves[sid]
        r, c = i // cols + 1, i % cols + 1
        fig.add_trace(go.Scatter(x=curve["exit_ts"], y=curve["equity"], mode="lines", showlegend=False,
                                 line=dict(color=CATEGORICAL[0], width=1.5)), row=r, col=c)
    fig.update_layout(template=TEMPLATE, title=f"Equity Curves by Strategy ({len(ranked)} strategies, faceted)",
                      height=220 * rows_n, margin=dict(l=10, r=10, t=60, b=10))
    return fig


def comparison_bar_figure(results: dict[str, dict], metric: str = "net_pnl", title: str | None = None) -> go.Figure:
    items = sorted(((sid, r["summary"].get(metric, 0) or 0) for sid, r in results.items()), key=lambda kv: kv[1])
    colors = [STATUS["good"] if v > 0 else STATUS["critical"] for _, v in items]
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=[sid for sid, _ in items], orientation="h",
                           marker_color=colors))
    fig.update_layout(template=TEMPLATE, title=title or metric.replace("_", " ").title(),
                      height=max(300, 28 * len(items)), margin=dict(l=10, r=10, t=40, b=10))
    return fig


def pattern_performance_table(results: dict[str, dict]) -> pd.DataFrame:
    """Win rate / avg R per candlestick or chart pattern, pooled across every strategy's trades that
    recorded which pattern(s) fired (candlestick_at_level, chart_pattern_breakout)."""
    rows = []
    for sid, r in results.items():
        trades = r["trades"]
        if trades is None or trades.empty or "patterns" not in trades.columns:
            continue
        for pattern_str, group in trades.groupby("patterns"):
            if not pattern_str:
                continue
            for pattern in str(pattern_str).split(","):
                pattern = pattern.strip()
                if not pattern:
                    continue
                wins = (group["pnl_net"] > 0).sum()
                r_values = group["r_multiple"].dropna()
                rows.append({"pattern": pattern, "strategy": sid, "trades": len(group), "wins": wins,
                            "sum_r": r_values.sum(), "count_r": len(r_values)})
    if not rows:
        return pd.DataFrame(columns=["pattern", "trades", "win_rate", "avg_r"])
    df = pd.DataFrame(rows).groupby("pattern").agg(trades=("trades", "sum"), wins=("wins", "sum"),
                                                   sum_r=("sum_r", "sum"), count_r=("count_r", "sum"))
    df["win_rate"] = df["wins"] / df["trades"]
    df["avg_r"] = df["sum_r"] / df["count_r"].replace(0, pd.NA)
    return df[["trades", "win_rate", "avg_r"]].sort_values("trades", ascending=False).reset_index()


def pattern_table_figure(pattern_df: pd.DataFrame) -> go.Figure:
    if pattern_df.empty:
        fig = go.Figure()
        fig.update_layout(template=TEMPLATE, title="Pattern performance (no pattern-tagged trades yet)", height=100)
        return fig
    fig = go.Figure(go.Table(
        header=dict(values=["Pattern", "Trades", "Win Rate", "Avg R"], fill_color="#2a2a27",
                   font=dict(color=TEXT_PRIMARY, size=12), align="left"),
        cells=dict(values=[pattern_df["pattern"], pattern_df["trades"], pattern_df["win_rate"].map("{:.0%}".format),
                          pattern_df["avg_r"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "-")],
                  fill_color=SURFACE, font=dict(color=TEXT_PRIMARY, size=11), align="left", height=24)))
    fig.update_layout(template=TEMPLATE, title="Pattern Performance (pooled across strategies)",
                      margin=dict(l=10, r=10, t=40, b=10), height=80 + 26 * len(pattern_df))
    return fig


def build_html_report(results_root: Path, strategy_ids: list[str], output_path: Path,
                      version: str = "v1", benchmark: pd.DataFrame | None = None) -> Path:
    results = load_strategy_results(results_root, strategy_ids, version)
    figs = [
        summary_table_figure(results),
        equity_curves_figure(results, benchmark),
        comparison_bar_figure(results, "net_pnl", "Net P&L by Strategy"),
        comparison_bar_figure(results, "win_rate", "Win Rate by Strategy"),
        pattern_table_figure(pattern_performance_table(results)),
    ]
    parts = ['<html><head><meta charset="utf-8"><title>Algo System Backtest Report</title>',
            f'<style>body{{background:{SURFACE};color:{TEXT_PRIMARY};'
            'font-family:Segoe UI,Arial,sans-serif;margin:0;padding:24px}'
            'h1{font-weight:600} .chart{margin-bottom:32px}</style></head><body>',
            "<h1>Algo Trading System - Backtest Report</h1>",
            f"<p style='color:{TEXT_SECONDARY}'>Strategies: {len(results)} of {len(strategy_ids)} requested "
            f"(others had no results yet). Results are on <b>v1, unoptimized default parameters</b> - "
            f"this proves the pipeline end to end, not that any strategy is ready to trade. "
            f"Phase 3's parameter sweeps/walk-forward come next.</p>"]
    for i, fig in enumerate(figs):
        parts.append(f'<div class="chart">{fig.to_html(full_html=False, include_plotlyjs=(i == 0))}</div>')
    parts.append("</body></html>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")
    return output_path
