"""Performance metrics, computed centrally from trade records - never inside a strategy. A strategy's
job is to emit signals; whether those signals turned out profitable (Sharpe, alpha, drawdown, ...) is
measured here, once, the same way for every mode (backtest, walk-forward, full-system, paper, live) -
they all produce trades with the same pnl_gross/costs/pnl_net shape, just two different class names
(SimTrade for backtests, ClosedTrade for live/paper) with the exit timestamp under different attribute
names (`exit_ts` vs `closed_at`) - `_exit_timestamp` is the one place that difference is handled.

Alpha/beta need a daily benchmark return series - see `benchmark_daily_returns` (NIFTY 50,
backtest_data/clean/1d/IDX_NSEI.parquet).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime

import pandas as pd

TRADING_DAYS_PER_YEAR = 252


@dataclass
class PerformanceMetrics:
    trades: int
    win_rate: float
    gross_pnl: float
    total_costs: float
    net_pnl: float
    cost_drag_pct: float          # total_costs / |gross_pnl| * 100 - how much of the edge fees ate
    profit_factor: float | None    # None when there are no losing trades to divide by (not JSON-safe as inf)
    expectancy: float             # average net P&L per trade
    max_drawdown: float           # on the net equity curve, in rupees (<= 0)
    sharpe: float | None          # annualized, net daily returns
    sortino: float | None
    cagr: float | None
    alpha: float | None            # annualized, vs a benchmark daily return series
    beta: float | None

    def as_dict(self) -> dict:
        return asdict(self)


EMPTY_METRICS = PerformanceMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None, None, None, None, None)


def _exit_timestamp(trade) -> datetime | None:
    return getattr(trade, "exit_ts", None) or getattr(trade, "closed_at", None)


def _as_naive_date(ts) -> pd.Timestamp:
    """Calendar date only - trades and the benchmark (see benchmark_daily_returns) must end up on the
    same tz-naive convention or pandas refuses to align/compare them at all (a real bug this one line
    fixes: real trades carry tz-aware IST timestamps, the old code left them tz-aware while the
    benchmark was already stripped, so alpha/beta crashed outside of tz-naive synthetic tests)."""
    t = pd.Timestamp(ts)
    return (t.tz_localize(None) if t.tzinfo is not None else t).normalize()


def _daily_net_pnl(trades: list) -> pd.Series:
    """Net P&L grouped by exit date - the unit Sharpe/Sortino/alpha/beta are computed over (trade-level
    returns aren't comparable to each other; daily portfolio returns are)."""
    rows = [(_as_naive_date(_exit_timestamp(t)), t.pnl_net) for t in trades if _exit_timestamp(t) is not None]
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series([r[1] for r in rows], index=pd.DatetimeIndex([r[0] for r in rows]))
    return s.groupby(level=0).sum().sort_index()


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    return float((equity - equity.cummax()).min())


def _sharpe(daily_returns: pd.Series) -> float | None:
    if len(daily_returns) < 2 or daily_returns.std(ddof=1) == 0:
        return None
    return float(daily_returns.mean() / daily_returns.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))


def _sortino(daily_returns: pd.Series) -> float | None:
    if len(daily_returns) < 2:
        return None
    downside = daily_returns[daily_returns < 0]
    if len(downside) < 2 or downside.std(ddof=1) == 0:
        return None
    return float(daily_returns.mean() / downside.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))


def _cagr(equity: pd.Series, starting_capital: float) -> float | None:
    if len(equity) < 2 or starting_capital <= 0:
        return None
    n_days = (equity.index[-1] - equity.index[0]).days
    total_return = equity.iloc[-1] / starting_capital
    if n_days <= 0 or total_return <= 0:
        return None
    return float(total_return ** (365.25 / n_days) - 1)


def benchmark_daily_returns(benchmark_close: pd.DataFrame) -> pd.Series:
    """benchmark_close: columns ts, close (e.g. a day's worth of IDX_NSEI bars). Simple daily returns,
    indexed by (timezone-naive, midnight-normalized) date, duplicates collapsed to the last value."""
    b = benchmark_close[["ts", "close"]].dropna().copy()
    ts = pd.to_datetime(b["ts"])
    b["ts"] = (ts.dt.tz_localize(None) if ts.dt.tz is not None else ts).dt.normalize()
    b = b.sort_values("ts").drop_duplicates("ts", keep="last").set_index("ts")["close"]
    return b.pct_change().dropna()


def _alpha_beta(daily_net_pnl: pd.Series, starting_capital: float, benchmark_returns: pd.Series
               ) -> tuple[float | None, float | None]:
    if starting_capital <= 0 or benchmark_returns.empty or len(daily_net_pnl) < 5:
        return None, None
    port = (daily_net_pnl / starting_capital).rename("p")
    aligned = pd.concat([port, benchmark_returns.rename("b")], axis=1).dropna()
    if len(aligned) < 5 or aligned["b"].var() == 0:
        return None, None
    beta = aligned["p"].cov(aligned["b"]) / aligned["b"].var()
    alpha_daily = aligned["p"].mean() - beta * aligned["b"].mean()
    return float(alpha_daily * TRADING_DAYS_PER_YEAR), float(beta)


def compute_metrics(trades: list, starting_capital: float = 1_000_000.0,
                    benchmark_returns: pd.Series | None = None) -> PerformanceMetrics:
    """`trades`: SimTrade (backtests) or ClosedTrade (paper/live) objects - anything with .pnl_gross,
    .costs, .pnl_net and an exit timestamp qualifies; no shared base class is required on purpose
    (backtest and live trade objects live in unrelated modules)."""
    if not trades:
        return EMPTY_METRICS
    pnl = [t.pnl_net for t in trades]
    wins, losses = [p for p in pnl if p > 0], [p for p in pnl if p <= 0]
    gross = sum(t.pnl_gross for t in trades)
    costs = sum(t.costs for t in trades)
    net = sum(pnl)
    gross_profit, gross_loss = sum(wins), -sum(losses)

    daily = _daily_net_pnl(trades)
    equity = starting_capital + daily.cumsum()
    daily_returns = daily / starting_capital if starting_capital > 0 else daily * 0.0
    alpha, beta = (None, None) if benchmark_returns is None else _alpha_beta(daily, starting_capital, benchmark_returns)

    return PerformanceMetrics(
        trades=len(trades), win_rate=len(wins) / len(trades), gross_pnl=gross, total_costs=costs, net_pnl=net,
        cost_drag_pct=(costs / abs(gross) * 100) if gross else 0.0,
        profit_factor=(gross_profit / gross_loss) if gross_loss > 0 else (None if gross_profit > 0 else 0.0),
        expectancy=net / len(trades), max_drawdown=_max_drawdown(equity),
        sharpe=_sharpe(daily_returns), sortino=_sortino(daily_returns), cagr=_cagr(equity, starting_capital),
        alpha=alpha, beta=beta)
