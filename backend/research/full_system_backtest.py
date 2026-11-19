"""Full-system backtest: every included strategy's signals, across every symbol, flowing through the
SAME Allocator and RiskManager objects Phase 4 built for live/paper trading - this is what proves
Phase 2 (the strategy pool) and Phase 4 (ensemble/risk/execution) actually work together, not just side
by side.

Not tick-by-tick replay (`scripts/paper_trade_replay.py`) - that reruns `build_features` on every closed
candle, which is fine for real live trading (5-minute gaps between candles) but far too slow for a
multi-month, multi-symbol backtest. Instead: each strategy's `generate_signals` scans its whole history
in one vectorized call (the same fast path `scripts/backtest_strategies.py` uses), and a position's exit
is resolved immediately via `backend.research.simple_backtester.scan_exit` (deterministic forward OHLC
scan - it doesn't need to "wait" for real time to find out where a trade would have exited).

What's NOT a shortcut: risk-manager state (kill switch, daily P&L, open positions/notional) and the
ensemble's weights must only reflect what was knowable at that real moment. So entries, real exits and
shadow exits are merged in one chronological event queue: a position opened on day 1 and still open on
day 5 counts against limits on day 3, and the performance tracker only learns a trade's outcome when
the walk reaches that trade's exit time - never earlier.

Live parity: entries fill at the NEXT bar's open (what StrategyRunner + OrderManager actually do - act on
a closed candle, fill on the next tick), stops gap-fill at the open when price jumps through them, and
every cost Angel One charges (incl. DP on delivery sells) is applied. Shadow tracking (every signal
measured as if taken alone) mirrors backend/ensemble/shadow_book.py in the live engine.
"""
from __future__ import annotations

import heapq
import itertools
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from datetime import date as date_type
from datetime import time

import pandas as pd

from backend.core.models import Horizon, Side, Signal
from backend.ensemble.allocator import Allocator
from backend.ensemble.performance_tracker import PerformanceTracker
from backend.execution import cost_model as cm
from backend.research.simple_backtester import BacktestResult, SimTrade, entry_fill, scan_exit
from backend.risk.risk_manager import RiskManager, not_executable_reason
from backend.strategies.base import Strategy, StrategyContext

log = logging.getLogger(__name__)

# Substrings of RiskManager reject reasons -> report bucket. Order matters (first match wins).
_REJECTION_BUCKETS = [
    ("kill switch", "kill_switch"), ("short selling", "delivery_short"), ("max concurrent", "max_positions"),
    ("no-entry window", "no_entry_window"), ("weight", "zero_weight"), ("stop already breached", "stop_breached"),
    ("fees would eat", "fees_too_high"), ("capped to zero", "capital_or_volume_cap"), ("zero", "degenerate_qty"),
]


def risk_rejection_bucket(reason: str) -> str:
    for needle, bucket in _REJECTION_BUCKETS:
        if needle in reason:
            return bucket
    return "other"


@dataclass
class FullSystemResult:
    trades: list[SimTrade] = field(default_factory=list)
    rejections: Counter = field(default_factory=Counter)   # bucket -> count; see _REJECTION_BUCKETS
    allocator_merges: int = 0     # (symbol, ts) groups where >1 strategy agreed and were combined
    shadow_samples: int = 0       # signal outcomes the performance tracker learned from

    def summary(self) -> dict:
        return BacktestResult(trades=self.trades).summary()

    def per_strategy_summary(self) -> dict[str, dict]:
        by_strategy: dict[str, list[SimTrade]] = defaultdict(list)
        for t in self.trades:
            by_strategy[t.strategy_id].append(t)
        return {sid: BacktestResult(trades=ts).summary() for sid, ts in by_strategy.items()}


@dataclass
class _Simulated:
    fill_price: float
    exit_ts: pd.Timestamp
    exit_price: float
    exit_reason: str
    pnl_gross: float
    costs: float
    pnl_net: float
    risk_amount: float


def _simulate(feat: pd.DataFrame, i: int, sig: Signal, qty: int, fill_price: float,
              square_off_time: time | None, max_holding_days: int, carry_days: int = 0) -> _Simulated:
    target = sig.targets[0] if sig.targets else None
    exit_ts, exit_price, exit_reason = scan_exit(feat, i, sig.side, sig.stop_loss, target, sig.horizon,
                                                 square_off_time, max_holding_days, carry_days)
    direction = 1 if sig.side == Side.BUY else -1
    pnl_gross = direction * (exit_price - fill_price) * qty
    overnight = pd.Timestamp(exit_ts).date() != pd.Timestamp(feat["ts"].iloc[i]).date()
    costs = cm.actual_trade_cost(fill_price, exit_price, qty, sig.side, sig.horizon, overnight).total
    atr = feat["atr_14"].iloc[i] if "atr_14" in feat.columns else None
    if atr is not None and pd.notna(atr):
        costs += cm.slippage(fill_price, float(atr), qty)
    return _Simulated(fill_price, exit_ts, exit_price, exit_reason, pnl_gross, costs, pnl_gross - costs,
                      qty * abs(fill_price - sig.stop_loss))


# Same gate and thresholds as backend/execution/strategy_runner.py's nifty50 BUY quality gate.
NIFTY50_TREND_TOP_PCT = 0.50       # raised from 0.35: more power to the trend-strength side of the gate
NIFTY50_LOSER_DROP_PCT = 0.05      # raised from 0.03: fewer names qualify as a contrarian "loser" bypass
NIFTY50_RS_LOOKBACK_DAYS = 126


def _build_nifty50_gate(feature_by_symbol_tf: dict[tuple[str, str], pd.DataFrame],
                        nifty50_symbols: set[str]) -> dict[date_type, tuple[set[str], set[str]]]:
    """Per calendar day, which nifty50 names are trend-aligned+top relative-strength or a sharp prior-day
    loser - computed only from daily bars closed ON OR BEFORE that day, so a later lookup for day D+1
    never sees data from day D+1 itself (no lookahead)."""
    series = {}
    for sym in nifty50_symbols:
        daily = feature_by_symbol_tf.get((sym, "1d"))
        if daily is not None and not daily.empty:
            series[sym] = daily[["ts", "close"]].reset_index(drop=True)
    all_days = sorted({pd.Timestamp(ts).date() for df in series.values() for ts in df["ts"]})
    ptr = dict.fromkeys(series, 0)
    gate: dict[date_type, tuple[set[str], set[str]]] = {}
    for day in all_days:
        scored: list[tuple[str, float]] = []
        losers: set[str] = set()
        for sym, df in series.items():
            n, p = len(df), ptr[sym]
            while p < n and pd.Timestamp(df["ts"].iloc[p]).date() <= day:
                p += 1
            ptr[sym] = p
            if p < 200:
                continue
            closes = df["close"].iloc[:p]
            latest = float(closes.iloc[-1])
            sma50, sma100, sma200 = (closes.tail(k).mean() for k in (50, 100, 200))
            trend_aligned = latest > sma50 > sma100 > sma200
            lookback = min(NIFTY50_RS_LOOKBACK_DAYS, len(closes) - 1)
            rel_return = latest / float(closes.iloc[-1 - lookback]) - 1 if lookback > 0 else 0.0
            scored.append((sym, rel_return if trend_aligned else float("-inf")))
            if float(closes.iloc[-1]) / float(closes.iloc[-2]) - 1 <= -NIFTY50_LOSER_DROP_PCT:
                losers.add(sym)
        scored.sort(key=lambda sr: sr[1], reverse=True)
        top_n = max(1, round(len(nifty50_symbols) * NIFTY50_TREND_TOP_PCT))
        gate[day] = ({s for s, sc in scored[:top_n] if sc != float("-inf")}, losers)
    return gate


def run_full_system_backtest(feature_by_symbol_tf: dict[tuple[str, str], pd.DataFrame],
                             strategies_by_timeframe: dict[str, list[Strategy]],
                             risk_manager: RiskManager, allocator: Allocator, tracker: PerformanceTracker,
                             square_off_time: time | None = time(15, 15), max_holding_days: int = 90,
                             entry_fill_mode: str = "next_open", carry_days: int = 0,
                             nifty50_symbols: set[str] | None = None) -> FullSystemResult:
    """Quantity always comes from `risk_manager.evaluate(...)` (risk-% sizing x ensemble weight, capped by
    notional/exposure/volume, rejected if fees would eat the edge) - never a fixed capital-per-trade."""
    result = FullSystemResult()

    # 1. Every strategy's full, vectorized signal list for every symbol/timeframe it applies to.
    #    Signal.timeframe isn't reliably set by every strategy - stamp it so the feature lookup works.
    all_signals: list[Signal] = []
    for (symbol, timeframe), feat in feature_by_symbol_tf.items():
        for strat in strategies_by_timeframe.get(timeframe, []):
            for sig in strat.run(StrategyContext(symbol, timeframe, feat)):
                sig.timeframe = timeframe
                all_signals.append(sig)

    # 2. Group every strategy's opinion for the same (symbol, moment) so the allocator sees them together.
    groups: dict[tuple[str, pd.Timestamp], list[Signal]] = defaultdict(list)
    for sig in all_signals:
        groups[(sig.symbol, sig.ts)].append(sig)
    sorted_keys = sorted(groups.keys(), key=lambda k: k[1])
    # ts -> row position per frame, built once (a `df["ts"] == ts` scan per signal is O(n) each)
    pos_index = {key: {t: i for i, t in enumerate(feat["ts"])} for key, feat in feature_by_symbol_tf.items()}

    # 3. One chronological walk over entries + real exits + shadow exits.
    open_positions: dict[str, tuple[SimTrade, float]] = {}   # symbol -> (resolved trade, notional)
    events: list[tuple] = []          # min-heap of (exit_ts, seq, kind, payload)
    seq = itertools.count()
    shadow_busy_until: dict[tuple[str, str], pd.Timestamp] = {}   # (strategy_id, symbol) -> exit ts
    current_day: date_type | None = None
    gi = 0

    nifty50_symbols = nifty50_symbols or set()
    nifty50_gate = _build_nifty50_gate(feature_by_symbol_tf, nifty50_symbols) if nifty50_symbols else {}
    nifty50_gate_days = sorted(nifty50_gate)
    if not nifty50_gate_days:
        nifty50_symbols = set()   # no daily data wired in for this run - don't silently block every BUY
    nifty50_gate_ptr = 0
    nifty50_trending: set[str] = set()
    nifty50_losers: set[str] = set()

    while gi < len(sorted_keys) or events:
        next_entry_ts = sorted_keys[gi][1] if gi < len(sorted_keys) else None
        next_exit_ts = events[0][0] if events else None
        take_exit = next_exit_ts is not None and (next_entry_ts is None or next_exit_ts <= next_entry_ts)
        event_ts = next_exit_ts if take_exit else next_entry_ts

        day = event_ts.date()
        if day != current_day:
            risk_manager.state.reset_for_new_day(day)
            current_day = day
            while nifty50_gate_ptr < len(nifty50_gate_days) and nifty50_gate_days[nifty50_gate_ptr] < day:
                nifty50_trending, nifty50_losers = nifty50_gate[nifty50_gate_days[nifty50_gate_ptr]]
                nifty50_gate_ptr += 1

        if take_exit:
            _, _, kind, payload = heapq.heappop(events)
            if kind == "shadow":
                strategy_id, regime, r_net = payload
                tracker.record(strategy_id, regime, r_net)
                result.shadow_samples += 1
            else:
                symbol = payload
                trade, notional = open_positions.pop(symbol)
                risk_manager.on_position_closed(Horizon(trade.horizon), notional)
                risk_manager.on_trade_closed(trade.pnl_net)
                result.trades.append(trade)
            continue

        symbol, ts = sorted_keys[gi]
        gi += 1
        sigs = groups[(symbol, ts)]

        # Shadow-track every signal (one shadow position per strategy/symbol at a time), executed or not.
        for sig in sigs:
            _schedule_shadow(sig, ts, feature_by_symbol_tf, pos_index, risk_manager, shadow_busy_until, events,
                             seq, square_off_time, max_holding_days, entry_fill_mode, carry_days)

        if symbol in open_positions:
            continue   # already in a trade for this symbol - same rule simulate_signals enforces

        allocated = allocator.decide(sigs)
        if not allocated:
            result.rejections[f"allocator_{allocator.last_drop_reason or 'dropped'}"] += 1
            continue
        chosen = allocated[0]
        if sum(1 for s in sigs if s.side == chosen.signal.side) > 1:
            result.allocator_merges += 1

        sig = chosen.signal
        if (sig.side == Side.BUY and symbol in nifty50_symbols
                and symbol not in nifty50_trending and symbol not in nifty50_losers):
            result.rejections["nifty50_not_trending"] += 1
            continue
        feat = feature_by_symbol_tf.get((symbol, sig.timeframe))
        if feat is None:
            continue
        i = pos_index[(symbol, sig.timeframe)].get(ts)
        if i is None:
            continue
        fill_price = entry_fill(feat, i, sig, entry_fill_mode)
        if fill_price is None:
            result.rejections["no_fill_bar"] += 1
            continue
        bar_volume = float(feat["volume"].iloc[i]) if "volume" in feat.columns else None
        atr = feat["atr_14"].iloc[i] if "atr_14" in feat.columns else None
        atr = float(atr) if atr is not None and pd.notna(atr) else None

        decision = risk_manager.evaluate(sig, ts, current_price=fill_price, weight=chosen.weight,
                                         bar_volume=bar_volume, atr=atr)
        if not decision.approved:
            result.rejections[risk_rejection_bucket(decision.reject_reason or "")] += 1
            continue

        sig = replace(sig, stop_loss=decision.stop_loss)   # may be wider than sig's own stop - see
                                                            # RiskManager.evaluate/widen_stop_for_cost_floor
        sim = _simulate(feat, i, sig, decision.qty, fill_price, square_off_time, max_holding_days, carry_days)
        target = sig.targets[0] if sig.targets else None
        trade = SimTrade(strategy_id=sig.strategy_id, strategy_version=sig.strategy_version, symbol=symbol,
                         horizon=sig.horizon.value, side=sig.side.value, entry_ts=ts, entry_price=fill_price,
                         exit_ts=sim.exit_ts, exit_price=sim.exit_price, qty=decision.qty,
                         stop_loss=sig.stop_loss, target=target, exit_reason=sim.exit_reason,
                         pnl_gross=sim.pnl_gross, costs=sim.costs, pnl_net=sim.pnl_net,
                         r_multiple=(sim.pnl_gross / sim.risk_amount) if sim.risk_amount > 0 else None,
                         patterns=",".join(sig.patterns), reason=sig.reason, regime=sig.regime)
        open_positions[symbol] = (trade, decision.notional)
        heapq.heappush(events, (sim.exit_ts, next(seq), "real", symbol))
        risk_manager.on_position_opened(sig.horizon, decision.notional)

    return result


def _schedule_shadow(sig: Signal, ts: pd.Timestamp, feature_by_symbol_tf, pos_index, risk_manager: RiskManager,
                     busy_until: dict, events: list, seq, square_off_time, max_holding_days, entry_fill_mode,
                     carry_days: int = 0) -> None:
    """Measure what this signal alone would have netted, in R, and queue that result to reach the
    performance tracker only at the trade's exit time. Signals the market wouldn't let you place (delivery
    shorts, no next bar, stop already breached at the fill) aren't measured at all."""
    key = (sig.strategy_id, sig.symbol)
    if key in busy_until and ts <= busy_until[key]:
        return
    if not_executable_reason(sig, risk_manager.limits):
        return
    feat = feature_by_symbol_tf.get((sig.symbol, sig.timeframe))
    if feat is None:
        return
    i = pos_index[(sig.symbol, sig.timeframe)].get(ts)
    if i is None:
        return
    fill_price = entry_fill(feat, i, sig, entry_fill_mode)
    if fill_price is None:
        return
    if (sig.side == Side.BUY and sig.stop_loss >= fill_price) or (sig.side == Side.SELL and sig.stop_loss <= fill_price):
        return
    qty = risk_manager.nominal_qty(sig, fill_price)
    if qty <= 0:
        return
    sim = _simulate(feat, i, sig, qty, fill_price, square_off_time, max_holding_days, carry_days)
    if sim.risk_amount <= 0:
        return
    busy_until[key] = sim.exit_ts
    heapq.heappush(events, (sim.exit_ts, next(seq), "shadow", (sig.strategy_id, sig.regime, sim.pnl_net / sim.risk_amount)))
