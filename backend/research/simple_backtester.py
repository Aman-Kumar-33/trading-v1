"""Phase 3's first backtester: turns a strategy's Signals into simulated Trades with realistic costs.

This is deliberately the *minimal* engine needed to get every strategy backtested honestly - not the
full event-driven engine with liquidity-based fill modeling, walk-forward, Monte-Carlo etc. described in
the plan's Phase 3 (that's the next, larger build-out). What it does do properly:
  - reuses the exact same Signal objects the strategy pool already produces (no separate "backtest-only"
    strategy logic to drift out of sync with live)
  - one open position per symbol at a time (a second signal for a symbol already in a trade is skipped,
    same as a live risk manager would reject it)
  - exit priority within a bar is stop-loss first, then target (the conservative assumption when both
    could plausibly have been hit - we don't know intra-bar path order from OHLC alone)
  - intraday positions are force-closed at the configured square-off time; long-term positions time out
    after `max_holding_days` if neither stop nor target is hit
  - every trade's cost is computed via backend.execution.cost_model: brokerage/STT/exchange/GST/stamp
    duty (`round_trip_cost`) plus a simple ATR-based slippage estimate (`slippage`, when `atr_14` is
    present in `df`) - fills are not assumed to happen at the exact theoretical price
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time

import pandas as pd

from backend.core.models import Horizon, Side, Signal
from backend.execution import cost_model as cm
from backend.risk.risk_manager import RiskLimits, pre_trade_check


@dataclass
class SimTrade:
    strategy_id: str
    strategy_version: str
    symbol: str
    horizon: str
    side: str
    entry_ts: pd.Timestamp
    entry_price: float
    exit_ts: pd.Timestamp
    exit_price: float
    qty: int
    stop_loss: float
    target: float | None
    exit_reason: str            # STOP | TARGET | SQUARE_OFF | TIMEOUT | END_OF_DATA
    pnl_gross: float
    costs: float
    pnl_net: float
    r_multiple: float | None
    patterns: str = ""
    reason: str = ""
    regime: str | None = None

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["entry_ts"] = str(self.entry_ts)
        d["exit_ts"] = str(self.exit_ts)
        return d


@dataclass
class BacktestResult:
    trades: list[SimTrade] = field(default_factory=list)

    def equity_curve(self, starting_capital: float = 0.0) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame(columns=["exit_ts", "pnl_net", "equity"])
        rows = sorted(((t.exit_ts, t.pnl_net) for t in self.trades), key=lambda r: r[0])
        df = pd.DataFrame(rows, columns=["exit_ts", "pnl_net"])
        df["equity"] = starting_capital + df["pnl_net"].cumsum()
        return df

    def summary(self) -> dict:
        if not self.trades:
            return {"trades": 0}
        pnl = [t.pnl_net for t in self.trades]
        wins = [p for p in pnl if p > 0]
        losses = [p for p in pnl if p <= 0]
        gross_profit = sum(wins)
        gross_loss = -sum(losses)
        equity = self.equity_curve()["equity"]
        running_max = equity.cummax()
        drawdown = (equity - running_max)
        r_values = [t.r_multiple for t in self.trades if t.r_multiple is not None]
        return {
            "trades": len(self.trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(self.trades),
            "gross_pnl": sum(t.pnl_gross for t in self.trades),
            "total_costs": sum(t.costs for t in self.trades),
            "net_pnl": sum(pnl),
            "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0,
            "expectancy": sum(pnl) / len(pnl),
            "avg_r": (sum(r_values) / len(r_values)) if r_values else None,
            "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        }


ENTRY_FILL_MODES = ("signal", "next_open")


def entry_fill(df: pd.DataFrame, i: int, sig: Signal, mode: str = "signal") -> float | None:
    """Price a signal on bar `i` actually gets filled at, or None if it can't be filled.
      "signal"    - at `sig.entry` (or bar i's close) on bar i itself: the classic research shortcut, and
                    slightly optimistic - the signal is computed FROM bar i's close, so you can't really
                    trade at that same close.
      "next_open" - at bar i+1's open: what the live system actually does (StrategyRunner acts when a
                    candle closes, OrderManager's market order fills on the next tick). An intraday
                    signal on the day's last bar can't fill (the next bar is tomorrow's session).
    Either way the position's exit scan starts at bar i+1, so `scan_exit(df, i, ...)` is correct for
    both: with next_open the fill is at bar i+1's open and that whole bar's range comes after it."""
    if mode not in ENTRY_FILL_MODES:
        raise ValueError(f"entry fill mode must be one of {ENTRY_FILL_MODES}, got {mode!r}")
    if i + 1 >= len(df):
        return None
    if mode == "signal":
        return float(sig.entry) if sig.entry is not None else float(df["close"].iloc[i])
    if sig.horizon == Horizon.INTRADAY and df["ts"].iloc[i + 1].date() != df["ts"].iloc[i].date():
        return None
    return float(df["open"].iloc[i + 1])


def scan_exit(df: pd.DataFrame, entry_i: int, side: Side, stop_loss: float, target: float | None,
             horizon: Horizon, square_off_time: time | None = time(15, 15),
             max_holding_days: int = 90, carry_days: int = 0) -> tuple[pd.Timestamp, float, str]:
    """Find the first bar after `entry_i` that closes a position opened on that bar - stop-loss first,
    then target (the conservative assumption when both could plausibly have been hit within the same
    bar - OHLC alone can't tell us intra-bar path order), else a forced intraday square-off or long-term
    timeout, else end of data. Shared by `simulate_signals` (one strategy/symbol at a time) and
    `backend.research.full_system_backtest` (every symbol interleaved in real time order) so the exit
    rule is identical in both - only how a position gets opened differs between them.

    A stop is filled at the stop price unless the bar OPENED beyond it (an overnight/news gap), in which
    case it fills at that worse open - a stop-market order can't fill at a price the market never
    traded. Targets stay at the target price (no credit for gapping through in your favour).

    `carry_days` > 0: an intraday LONG isn't force-closed at square-off - it's carried (as delivery) and
    exits on its original stop/target, or times out `carry_days` calendar days after entry. Intraday
    shorts can't be carried in the NSE cash segment, so they always square off the same day."""
    carried_long = horizon == Horizon.INTRADAY and side == Side.BUY and carry_days > 0
    intraday = horizon == Horizon.INTRADAY and not carried_long
    ts = df["ts"].array   # O(1) element access; df.iloc[j] builds a whole Series per bar (~10x slower)
    lows, highs, closes = df["low"].to_numpy(), df["high"].to_numpy(), df["close"].to_numpy()
    opens = df["open"].to_numpy() if "open" in df.columns else None
    entry_day = ts[entry_i].date()
    hold_days = carry_days if carried_long else max_holding_days
    deadline_day = entry_day if intraday else (entry_day + pd.Timedelta(days=hold_days))
    for j in range(entry_i + 1, len(df)):
        ts_j = ts[j]
        row_day = ts_j.date()
        if intraday and row_day != entry_day:
            return ts[j - 1], float(closes[j - 1]), "END_OF_DATA"
        if intraday and square_off_time is not None and ts_j.time() >= square_off_time:
            return ts_j, float(closes[j]), "SQUARE_OFF"
        if not intraday and row_day > deadline_day:
            return ts[j - 1], float(closes[j - 1]), "TIMEOUT"
        stop_hit = lows[j] <= stop_loss if side == Side.BUY else highs[j] >= stop_loss
        if stop_hit:
            fill = float(stop_loss)
            if opens is not None:
                fill = min(fill, float(opens[j])) if side == Side.BUY else max(fill, float(opens[j]))
            return ts_j, fill, "STOP"
        if target is not None:
            target_hit = highs[j] >= target if side == Side.BUY else lows[j] <= target
            if target_hit:
                return ts_j, float(target), "TARGET"
    return ts[len(df) - 1], float(closes[-1]), "END_OF_DATA"


def simulate_signals(df: pd.DataFrame, signals: list[Signal], capital_per_trade: float = 20_000.0,
                     square_off_time: time | None = time(15, 15), max_holding_days: int = 90,
                     entry_fill_mode: str = "signal", limits: RiskLimits | None = None,
                     carry_days: int = 0) -> BacktestResult:
    """`df` must be indexed positionally 0..n-1 with a `ts` column matching the timestamps signals were
    generated from (the same DataFrame passed to the strategy). Signals are processed in timestamp
    order; a signal for a symbol that's already in an open position is skipped. See `entry_fill` for
    what `entry_fill_mode` means - "next_open" matches the live system, "signal" is the old default.
    With `limits`, every trade must also pass backend.risk.risk_manager.pre_trade_check (no delivery
    shorts, fee hurdles) - the same per-trade rules live applies, so research doesn't credit trades live
    would never take."""
    result = BacktestResult()
    if not signals:
        return result
    ts_to_pos = {t: i for i, t in enumerate(df["ts"])}
    ordered = sorted(signals, key=lambda s: s.ts)
    open_until_ts: pd.Timestamp | None = None   # this symbol is busy until this timestamp

    for sig in ordered:
        if open_until_ts is not None and sig.ts <= open_until_ts:
            continue
        i = ts_to_pos.get(sig.ts)
        if i is None:
            continue
        entry_price = entry_fill(df, i, sig, entry_fill_mode)
        if entry_price is None:
            continue
        qty = max(1, int(capital_per_trade // entry_price))
        atr_i = df["atr_14"].iloc[i] if "atr_14" in df.columns else None
        atr_i = float(atr_i) if atr_i is not None and pd.notna(atr_i) else None
        if limits is not None and pre_trade_check(sig, entry_price, qty, limits, atr_i)[0]:
            continue
        target = sig.targets[0] if sig.targets else None
        risk_per_share = abs(entry_price - sig.stop_loss)

        exit_ts, exit_price, exit_reason = scan_exit(df, i, sig.side, sig.stop_loss, target, sig.horizon,
                                                      square_off_time, max_holding_days, carry_days)

        direction = 1 if sig.side == Side.BUY else -1
        pnl_gross = direction * (exit_price - entry_price) * qty
        overnight = pd.Timestamp(exit_ts).date() != pd.Timestamp(df["ts"].iloc[i]).date()
        costs = cm.actual_trade_cost(entry_price, exit_price, qty, sig.side, sig.horizon, overnight).total
        entry_atr = df["atr_14"].iloc[i] if "atr_14" in df.columns else None
        if entry_atr is not None and pd.notna(entry_atr):
            costs += cm.slippage(entry_price, float(entry_atr), qty)
        pnl_net = pnl_gross - costs
        r_multiple = (pnl_gross / qty) / risk_per_share if risk_per_share > 0 else None

        result.trades.append(SimTrade(
            strategy_id=sig.strategy_id, strategy_version=sig.strategy_version, symbol=sig.symbol,
            horizon=sig.horizon.value, side=sig.side.value, entry_ts=sig.ts, entry_price=entry_price,
            exit_ts=exit_ts, exit_price=exit_price, qty=qty, stop_loss=sig.stop_loss, target=target,
            exit_reason=exit_reason, pnl_gross=pnl_gross, costs=costs, pnl_net=pnl_net,
            r_multiple=r_multiple, patterns=",".join(sig.patterns), reason=sig.reason))
        open_until_ts = exit_ts

    return result
