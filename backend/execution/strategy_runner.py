"""Simplified live wiring for the second (sandbox) system: on every closed candle, rebuild features and run
the MACD strategy - no allocator, no risk manager gating, no nifty50/breadth filters. Each buy is sized by
backend/execution/sizing.py (5-25% of capital, decided per signal) and at most `max_open_positions` longs are held
at once - a BUY while full is skipped, not queued. A RiskManager instance is still passed to OrderManager
only as its bookkeeping sink (on_position_opened/closed) - its `evaluate()` gate is never called, by
design (the user explicitly wants no risk layer here).

The strategy is long/flat. A BUY signal opens a long when flat; a SELL signal only ever CLOSES a long -
it never opens a short. Besides acting on the cross itself, every closed bar also checks "is MACD below
its signal line while we hold a long?" and exits if so: that is the same exit a missed bearish cross
(restart, feed gap, a cross on the 15:15 closing bar) would have produced, never a new rule.

Stock list: only symbols on the watchlist (set_watchlist) are MACD-checked and bought; everything else is
ignored. A stock that already has an open position stays checked even if it is later removed from the list,
but only for its exit - it never gets a new BUY - so a position is never left without a sell rule.

Every signal produces a Telegram message saying what the system did with it (bought / sold / ignored and why).

Signal mode "live" (the 5 minute system): MACD is not only read when a candle closes. Every tick of a tracked stock
updates the forming candle's MACD (the latest price taken as its close - what a live chart draws) from the last
closed bar's EMA values, and a BUY fires the moment MACD is more than `tolerance` above the signal line after
being below it (a SELL the moment it is more than `tolerance` below, while a long is held). Between -tolerance
and +tolerance nothing changes, so small wobbles around the line are ignored. The tolerance is max(`tolerance`,
`tolerance_atr` x the stock's ATR): a fixed number is far too small for a Rs 12,000 stock and about right for a Rs 300 one,
so it is scaled to each stock's own volatility. The cross must also hold for `hold_seconds` before the order is placed -
most flips around the line last only a few seconds.

Signal mode "candle_confirmed" (the 5 minute system): signals come only from COMPLETED candles. A BUY needs the MACD line to
cross above its signal line and then be more than the tolerance above it on a closed bar (the first closed bar beyond the
tolerance after the lines were last at/below each other - see `confirmed_cross_up`). A SELL is the next plain intersection:
the first closed bar on which MACD is below its signal line, with no tolerance. Every BUY gets a stop `stop_loss_pct`
below the fill price (OrderManager) and none is taken before `trade_start`.

Series continuity: a symbol is only evaluated once its warm-up history has been loaded from Angel One
(`set_history`). The 15:15-15:30 closing bar is part of the series (broker charts have it, and MACD
depends on every bar) but nothing can be traded after 15:30, so a signal on that bar is carried over and
executed on the first tick of the next session.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from backend.core import events
from backend.core.events import EventBus
from backend.core.models import Candle, Horizon, SessionEvent, SessionState, Side, Signal, Tick
from backend.core.timeframes import minutes
from backend.data.macd_recorder import MacdRecorder
from backend.execution import sizing
from backend.execution.order_manager import OrderManager
from backend.features import indicators as ind
from backend.features.pipeline import build_features
from backend.notify import telegram as tg
from backend.strategies.base import Strategy, StrategyContext
from backend.strategies.intraday.macd_only import confirmed_cross_up  # noqa: F401 - the entry rule of "candle_confirmed" mode

log = logging.getLogger(__name__)

MIN_BARS_TO_RUN = 60
SESSION_OPEN, SESSION_CLOSE = time(9, 15), time(15, 30)
CLOSING_BAR_START = time(15, 15)   # the short 15:15-15:30 bar - complete only when the market has shut
STATE_EXIT_IDS = {"macd_only"}     # strategies whose open longs also exit while MACD < signal
BAR_COLUMNS = ["ts", "open", "high", "low", "close", "volume"]

SignalSink = Callable[[Signal, str, str | None], None]   # (signal, status, reject_reason)


def _in_session(ts: datetime) -> bool:
    return ts.weekday() < 5 and SESSION_OPEN <= ts.time() < SESSION_CLOSE


def _bucket_start(ts: datetime, timeframe: str) -> datetime:
    """Start of the candle a timestamp falls in (candles are aligned to the 09:15 open)."""
    step = minutes(timeframe)
    open_min = SESSION_OPEN.hour * 60 + SESSION_OPEN.minute
    start = ((ts.hour * 60 + ts.minute - open_min) // step) * step + open_min
    return ts.replace(hour=start // 60, minute=start % 60, second=0, microsecond=0)


@dataclass
class _LiveMacd:
    """The EMA values after the newest CLOSED bar - the base every forming-candle MACD is computed from."""
    last_start: datetime        # start of that closed bar
    e12: float
    e26: float
    sig: float
    gap: float                  # MACD - signal on that closed bar
    atr: float | None
    state: str | None           # "up" / "down": the side of the signal line (beyond the tolerance) MACD was last on


class SimpleRunner:
    def __init__(self, bus: EventBus, strategies_by_timeframe: dict[str, list[Strategy]],
                 history_by_symbol_tf: dict[tuple[str, str], pd.DataFrame], order_manager: OrderManager,
                 capital: float = 1_000_000.0, history_bars: int = 500,
                 signal_sink: SignalSink | None = None, require_history: bool = True,
                 deferred_file: Path | None = None, watchlist: set[str] | None = None, notify_signals: bool = True,
                 min_position_pct: float = 0.05, default_position_pct: float = 0.10, max_position_pct: float = 0.25,
                 max_open_positions: int = 0, trend_scores: dict[str, float] | None = None,
                 recorder: MacdRecorder | None = None, square_off_time: time | None = None,
                 signal_mode: str = "bar_close", tolerance: float = 0.0, tolerance_atr: float = 0.0,
                 hold_seconds: float = 0.0, trade_start: time | None = None, stop_loss_pct: float | None = None):
        self.bus = bus
        # "bar_close" (main system) | "candle_confirmed" (5 minute system: closed candles, tolerance on the entry only) |
        # "live" (tick-driven, kept for research)
        self.signal_mode = signal_mode
        self.trade_start = trade_start        # no BUY before this time of day (exits are never blocked)
        self.stop_loss_pct = float(stop_loss_pct) if stop_loss_pct else None   # BUY stop = this share below the entry
        self.tolerance = float(tolerance)     # live mode: how far beyond the signal line MACD must be to count
        self._live: dict[str, _LiveMacd] = {}
        self._live_tried: dict[str, datetime] = {}      # symbol -> candle a (re)build was already attempted in
        self.tolerance_atr = float(tolerance_atr)  # live mode: tolerance as a multiple of the stock's ATR (a floor of `tolerance`)
        self.hold_seconds = float(hold_seconds)    # live mode: the cross must last this long before it is acted on
        self._pending: dict[str, tuple[Side, datetime]] = {}   # symbol -> (side, since when the cross has held)
        self._sell_retry_at: dict[str, datetime] = {}
        # Intraday-only mode: set to the square-off time and every position is closed then (OrderManager), so no BUY is
        # taken at/after it and a BUY from the 15:15 closing bar is dropped instead of carried to the next morning.
        self.square_off_time = square_off_time
        self.recorder = recorder   # research data for the listed stocks (ticks, bars, MACD analysis, decisions)
        self.strategies_by_timeframe = strategies_by_timeframe
        self.history_by_symbol_tf = history_by_symbol_tf
        self.order_manager = order_manager
        self.capital = capital
        self.min_position_pct, self.default_position_pct = min_position_pct, default_position_pct
        self.max_position_pct, self.max_open_positions = max_position_pct, max_open_positions
        self.trend_scores: dict[str, float] = dict(trend_scores or {})   # symbol -> 0-100, auto mode only
        self.history_bars = history_bars
        self.signal_sink = signal_sink
        # With require_history a (symbol, timeframe) is only evaluated after set_history() - the live run
        # needs that guarantee; tests that pass a ready-made history dict turn it off.
        self._ready: set[tuple[str, str]] = set() if require_history else set(history_by_symbol_tf)
        self._require_history = require_history
        self._today_bars: dict[tuple[str, str], list[Candle]] = defaultdict(list)
        self._evaluated: set[tuple[str, str, datetime]] = set()   # (symbol, timeframe, bar start) already run through the strategy
        self._deferred: dict[str, tuple[Signal, date]] = {}   # symbol -> (closing-bar signal, its trading date)
        self._deferred_ctx: dict[str, sizing.SizeContext] = {}   # what sizing needs, kept with the carried signal
        # the process exits at 15:35 and a new one starts next morning, so a carried-over signal has to
        # live on disk, not only in memory
        self._deferred_file = Path(deferred_file) if deferred_file else None
        self._load_deferred()
        self._warned_not_ready: set[tuple[str, str]] = set()
        self._watchlist: set[str] | None = set(watchlist) if watchlist is not None else None   # None = every symbol
        self.notify_signals = notify_signals
        bus.subscribe(events.CANDLE_CLOSED, self.on_candle_closed)
        bus.subscribe(events.TICK, self.on_tick)
        bus.subscribe(events.SESSION, self.on_session)

    # ------------------------------------------------------------------ history
    def set_history(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        self.history_by_symbol_tf[(symbol, timeframe)] = df
        self._ready.add((symbol, timeframe))

    def is_ready(self, symbol: str, timeframe: str) -> bool:
        return (symbol, timeframe) in self._ready

    # ------------------------------------------------------------------ stock list
    def set_watchlist(self, symbols) -> None:
        self._watchlist = set(symbols)

    def set_trend_scores(self, scores: dict[str, float]) -> None:
        self.trend_scores = dict(scores)

    def _is_watched(self, symbol: str) -> bool:
        return self._watchlist is None or symbol in self._watchlist

    def _is_tracked(self, symbol: str) -> bool:
        return self._is_watched(symbol) or self.order_manager.has_position(symbol)

    # ------------------------------------------------------------------ events
    async def on_session(self, ev: SessionEvent) -> None:
        if ev.state == SessionState.OPEN and ev.trading_date:
            self.order_manager.risk_manager.state.reset_for_new_day(date.fromisoformat(ev.trading_date))

    async def on_tick(self, tick: Tick) -> None:
        if self.recorder is not None and self._is_watched(tick.symbol):
            self.recorder.record_tick(tick)
        await self.order_manager.on_tick(tick)
        await self.order_manager.on_time(tick.ts)
        if self.signal_mode == "live" and self._is_tracked(tick.symbol):
            await self._live_tick(tick)
        pending = self._deferred.get(tick.symbol)
        if pending is not None and _in_session(tick.ts) and tick.ts.date() > pending[1]:
            del self._deferred[tick.symbol]
            self._save_deferred()
            ctx = self._deferred_ctx.pop(tick.symbol, None)
            if self.square_off_time is not None and pending[0].side == Side.BUY:
                self._finish(pending[0], "REJECTED", "intraday only - cross was on the previous day's 15:15 bar",
                             "SKIPPED - intraday only: a buy from yesterday's closing bar is not taken the next morning")
                return
            await self._act(tick.symbol, pending[0], tick.ts, tick.ltp, note="carried over from the 15:15 closing bar", ctx=ctx)

    async def on_candle_closed(self, candle: Candle) -> None:
        tf = candle.timeframe
        # a candle's `end` is the next bucket's start; the closing 15:15 bar ends exactly at 15:30, so test
        # its START against the session instead
        if tf not in self.strategies_by_timeframe or not _in_session(candle.start):
            return
        key = (candle.symbol, tf)
        # Every bar is kept, even for stocks that are not on the list: in auto mode the whole index is
        # followed, and the stocks picked must already have their 09:15 bar in the series.
        self._today_bars[key].append(candle)
        if not self._is_tracked(candle.symbol):
            return   # not on the stock list and nothing to manage - bars kept, but never evaluated
        if self._require_history and key not in self._ready:
            if key not in self._warned_not_ready:
                self._warned_not_ready.add(key)
                log.warning("bar skipped - warm-up history not loaded yet for this symbol",
                            extra={"data": {"symbol": candle.symbol, "bar": candle.start.isoformat()}})
            return
        df = self._build_df(key)
        if len(df) < MIN_BARS_TO_RUN:
            return
        self._warn_on_gap(candle, df)
        feat = build_features(df, tf, core_only=True)
        latest_ts = feat["ts"].iloc[-1]
        if latest_ts != pd.Timestamp(candle.start):
            log.warning("closed bar is not the newest bar in the series - skipping",
                        extra={"data": {"symbol": candle.symbol, "bar": candle.start.isoformat(),
                                        "latest": str(latest_ts)}})
            return
        closing_bar = candle.start.time() >= CLOSING_BAR_START and candle.end.time() >= SESSION_CLOSE
        last = feat.iloc[-1]
        macd_state = {"bar": candle.start.isoformat(), "close": candle.close,
                      "macd": _f(last["macd"]), "signal": _f(last["macd_signal"]), "hist": _f(last["macd_hist"])}
        ctx = self._size_context(candle.symbol, feat)
        self._evaluated.add((candle.symbol, tf, candle.start))
        if self.signal_mode == "live":
            # signals come from the ticks (_live_tick); a closed bar only refreshes the base the next candle starts from
            self._set_live(candle.symbol, feat)
            if self.recorder is not None and self._is_watched(candle.symbol):
                self._record_bar(candle.symbol, feat, "live", candle)
            return
        for strat in self.strategies_by_timeframe[tf]:
            sigs = strat.run(StrategyContext(candle.symbol, tf, feat))
            if self.signal_mode == "candle_confirmed":
                # exits stay the plain cross; entries are replaced by the tolerance-confirmed cross
                sigs = [x for x in sigs if x.side == Side.SELL]
                buy = self._confirmed_buy(candle, feat, strat)
                if buy is not None:
                    sigs.append(buy)
            for sig in sigs:
                if sig.ts != latest_ts:
                    continue
                log.info("macd signal", extra={"data": {"symbol": candle.symbol, "side": sig.side.value,
                                                        "strategy": sig.strategy_id, **macd_state}})
                if closing_bar and self.square_off_time is None:
                    self._deferred[candle.symbol] = (sig, candle.start.date())
                    self._deferred_ctx[candle.symbol] = ctx
                    self._save_deferred()
                    self._finish(sig, "ACCEPTED", "closing 15:15 bar - market shut, executes at next open",
                                 "market is shut (this was the 15:15 closing bar) - will be executed at the next open")
                else:
                    await self._act(candle.symbol, sig, candle.end, candle.close, ctx=ctx)
            if not closing_bar and strat.strategy_id in STATE_EXIT_IDS:
                await self._exit_if_bearish(candle, strat.strategy_id, macd_state)
        if self.recorder is not None and self._is_watched(candle.symbol):
            self._record_bar(candle.symbol, feat, "live", candle)

    def _size_context(self, symbol: str, feat: pd.DataFrame) -> sizing.SizeContext:
        last = feat.iloc[-1]
        hist_jump = _f(last["macd_hist"] - feat["macd_hist"].iloc[-2]) if len(feat) > 1 else None
        return sizing.SizeContext(hist_jump=hist_jump, atr=_f(last["atr_14"]), trend_score=self.trend_scores.get(symbol))

    def _tolerance_for(self, atr: float | None) -> float:
        return max(self.tolerance, self.tolerance_atr * atr) if atr else self.tolerance

    def _confirmed_buy(self, candle: Candle, feat: pd.DataFrame, strat: Strategy) -> Signal | None:
        """The BUY of "candle_confirmed" mode, or None: the newest closed bar is the first beyond the tolerance after an
        intersection of the MACD and signal lines (MACD going from at/below to above)."""
        gap = (feat["macd"] - feat["macd_signal"]).to_numpy(dtype=float)
        atr = _f(feat["atr_14"].iloc[-1])
        tol = self._tolerance_for(atr)
        if not confirmed_cross_up(gap, tol):
            return None
        macd, sig = float(feat["macd"].iloc[-1]), float(feat["macd_signal"].iloc[-1])
        entry = float(candle.close)
        stop = entry * (1 - self.stop_loss_pct) if self.stop_loss_pct else entry - 10 * (atr if atr else entry * 0.02)
        return Signal(symbol=candle.symbol, side=Side.BUY, horizon=strat.horizon, stop_loss=stop, strategy_id=strat.strategy_id,
                      strategy_version=strat.version, ts=feat["ts"].iloc[-1], timeframe=candle.timeframe, entry=entry,
                      reason=f"MACD bullish cross confirmed on the closed bar (macd {macd:.4f} > signal {sig:.4f}, "
                             f"gap {macd - sig:+.4f} > tolerance {tol:.3f}; the lines were at/below each other before)")

    async def catch_up_entries(self, symbols, now: datetime) -> None:
        """A stock that is put on the list (auto mode picks before the open; a late restart or an edited list
        adds stocks later) or whose history finishes loading just AFTER a bar closed would miss that bar's cross for good.
        If the stock's newest closed bar is a fresh BUY cross that was never evaluated and closed less than one bar
        ago, take it now. Same signal, same sizing, executed up to a bar late instead of never. Bars the runner
        already evaluated are skipped, so nothing is bought twice."""
        if self.signal_mode == "live":
            return   # live mode acts on ticks as they arrive
        for sym in symbols:
            if not self._is_watched(sym):
                continue
            for tf, strats in self.strategies_by_timeframe.items():
                key = (sym, tf)
                bars = self._today_bars.get(key)
                if not bars or (self._require_history and key not in self._ready):
                    continue
                candle = bars[-1]
                if (sym, tf, candle.start) in self._evaluated or now - candle.end > timedelta(minutes=minutes(tf)):
                    continue
                df = self._build_df(key)
                if len(df) < MIN_BARS_TO_RUN:
                    continue
                feat = build_features(df, tf, core_only=True)
                if feat["ts"].iloc[-1] != pd.Timestamp(candle.start):
                    continue
                self._evaluated.add((sym, tf, candle.start))
                ctx = self._size_context(sym, feat)
                for strat in strats:
                    sigs = strat.run(StrategyContext(sym, tf, feat))
                    if self.signal_mode == "candle_confirmed":
                        buy = self._confirmed_buy(candle, feat, strat)
                        sigs = [buy] if buy is not None else []
                    for sig in sigs:
                        if sig.ts != feat["ts"].iloc[-1] or sig.side != Side.BUY:
                            continue
                        log.info("macd signal caught up - the stock joined the list after this bar closed",
                                 extra={"data": {"symbol": sym, "bar": candle.start.isoformat(), "strategy": sig.strategy_id}})
                        await self._act(sym, sig, now, candle.close,
                                        note="caught up: the cross closed just before this stock was on the list", ctx=ctx)

    # ------------------------------------------------------------------ live (tick-driven) MACD
    def _live_timeframe(self) -> str:
        return next(iter(self.strategies_by_timeframe))

    def _set_live(self, symbol: str, feat: pd.DataFrame) -> bool:
        macd, sig = feat["macd"].to_numpy(dtype=float), feat["macd_signal"].to_numpy(dtype=float)
        if len(feat) == 0 or np.isnan(macd[-1]) or np.isnan(sig[-1]):
            return False
        close = feat["close"].astype(float)
        gap = macd - sig
        decisive = np.flatnonzero(np.abs(gap) > self.tolerance)
        prev = self._live.get(symbol)
        state = prev.state if prev is not None and prev.state is not None else (
            None if not len(decisive) else ("up" if gap[decisive[-1]] > 0 else "down"))
        self._live[symbol] = _LiveMacd(
            last_start=pd.Timestamp(feat["ts"].iloc[-1]).to_pydatetime(),
            e12=float(ind.ema(close, 12).iloc[-1]), e26=float(ind.ema(close, 26).iloc[-1]), sig=float(sig[-1]),
            gap=float(gap[-1]), atr=_f(feat["atr_14"].iloc[-1]), state=state)
        return True

    def _rebuild_live(self, symbol: str, tf: str) -> bool:
        key = (symbol, tf)
        if self._require_history and key not in self._ready:
            return False
        df = self._build_df(key)
        if len(df) < MIN_BARS_TO_RUN:
            return False
        return self._set_live(symbol, build_features(df, tf, core_only=True))

    async def _live_tick(self, tick: Tick) -> None:
        tf = self._live_timeframe()
        if not _in_session(tick.ts):
            return
        sym, step = tick.symbol, timedelta(minutes=minutes(tf))
        bucket = _bucket_start(tick.ts, tf)

        def contiguous(base: _LiveMacd | None) -> bool:
            return base is not None and (bucket - base.last_start == step or
                                         (bucket.time() == SESSION_OPEN and base.last_start.date() < bucket.date()))
        base = self._live.get(sym)
        if not contiguous(base):
            # the base must be the bar right before the forming one; otherwise rebuild it from the series (once per
            # candle) - or wait for the bar-close event, which sets it
            if self._live_tried.get(sym) == bucket:
                return
            self._live_tried[sym] = bucket
            self._rebuild_live(sym, tf)
            base = self._live.get(sym)
            if not contiguous(base):
                return
        a12, a26, a9 = 2 / 13, 2 / 27, 2 / 10
        price = tick.ltp
        e12, e26 = a12 * price + (1 - a12) * base.e12, a26 * price + (1 - a26) * base.e26
        macd = e12 - e26
        sig = a9 * macd + (1 - a9) * base.sig
        gap = macd - sig
        tol = max(self.tolerance, self.tolerance_atr * base.atr) if base.atr else self.tolerance
        strat = self.strategies_by_timeframe[tf][0]
        holding = self.order_manager.has_position(sym, strat.strategy_id)
        want = None
        if gap > tol:
            if base.state == "down" and not holding:
                want = Side.BUY
            else:
                base.state = "up"
        elif gap < -tol:
            if holding or base.state == "up":
                want = Side.SELL          # (with nothing held it is reported like any other signal and sells nothing)
            else:
                base.state = "down"
        if want is None:
            self._pending.pop(sym, None)  # back inside the band: the cross did not hold
            return
        pend = self._pending.get(sym)
        since = pend[1] if pend is not None and pend[0] == want else tick.ts
        self._pending[sym] = (want, since)
        if (tick.ts - since).total_seconds() < self.hold_seconds:
            return                        # crossed, but not for long enough yet
        if want == Side.SELL and holding and self._sell_retry_at.get(sym, tick.ts) > tick.ts:
            return                        # the last exit order failed - wait before trying again
        self._pending.pop(sym, None)
        side = want
        base.state = "up" if side == Side.BUY else "down"
        if side == Side.SELL and holding:
            self._sell_retry_at[sym] = tick.ts + timedelta(seconds=10)
        buy = side == Side.BUY
        entry = float(price)
        atr = base.atr if base.atr else entry * 0.02
        sig_obj = Signal(symbol=sym, side=side, horizon=strat.horizon,
                         stop_loss=entry - 10 * atr if buy else entry + 10 * atr,
                         strategy_id=strat.strategy_id, strategy_version=strat.version, ts=tick.ts, timeframe=tf, entry=entry,
                         reason=f"MACD live {'bullish' if buy else 'bearish'} cross (macd {macd:.4f} {'>' if buy else '<'} "
                                f"signal {sig:.4f}, gap {gap:+.4f}, tolerance {tol:.3f}, held {(tick.ts - since).total_seconds():.0f}s)")
        log.info("macd signal (live)", extra={"data": {"symbol": sym, "side": side.value, "strategy": strat.strategy_id,
                                                         "price": entry, "macd": macd, "signal": sig, "gap": gap,
                                                         "tolerance": tol, "held_seconds": (tick.ts - since).total_seconds(),
                                                         "bar": bucket.isoformat()}})
        ctx = sizing.SizeContext(hist_jump=gap - base.gap, atr=base.atr, trend_score=self.trend_scores.get(sym))
        await self._act(sym, sig_obj, tick.ts, entry, ctx=ctx)

    # ------------------------------------------------------------------ research recording
    def _record_bar(self, symbol: str, feat: pd.DataFrame, source: str, candle: Candle | None = None,
                    only_day: bool = False) -> None:
        score = self.trend_scores.get(symbol)
        rank = 1 + sum(1 for s in self.trend_scores.values() if s > score) if score is not None else None
        self.recorder.record_bars(symbol, feat, only_day, source, live_candle=candle,
                                  holding=self.order_manager.has_position(symbol), trend_score=score, pick_rank=rank)

    def backfill_recording(self, symbols) -> None:
        """Write today's bars that closed before a stock was on the list (added after the open) or before its
        history had loaded - rebuilt from the same series the strategy uses. Safe to call repeatedly."""
        if self.recorder is None:
            return
        for sym in symbols:
            for tf in self.strategies_by_timeframe:
                key = (sym, tf)
                if not self._is_watched(sym) or (self._require_history and key not in self._ready):
                    continue
                df = self._build_df(key)
                if len(df) >= MIN_BARS_TO_RUN:
                    self._record_bar(sym, build_features(df, tf, core_only=True), "backfill", only_day=True)

    # ------------------------------------------------------------------ acting on signals
    async def _exit_if_bearish(self, candle: Candle, strategy_id: str, state: dict) -> None:
        macd, sig = state["macd"], state["signal"]
        if macd is None or sig is None or not macd < sig:
            return
        if not self.order_manager.has_position(candle.symbol, strategy_id):
            return
        closed = await self.order_manager.close_on_signal(candle.symbol, Side.SELL, candle.end, strategy_id=strategy_id)
        if closed:
            log.info("long closed - MACD is below its signal line (catch-up exit, no fresh cross this bar)",
                     extra={"data": {"symbol": candle.symbol, "strategy": strategy_id, **state}})
            if self.recorder is not None:
                self.recorder.note_decision(candle.symbol, candle.start, strategy_id, "SELL", candle.timeframe, candle.close,
                                            "ACCEPTED", "catch-up exit - MACD below signal, no fresh cross this bar",
                                            f"SOLD {sum(t.position.qty for t in closed)} @ {closed[0].exit_price:,.2f}")

    async def _act(self, symbol: str, sig: Signal, now: datetime, price: float, note: str = "",
                   ctx: sizing.SizeContext | None = None) -> None:
        has_position = self.order_manager.has_position(symbol, sig.strategy_id)
        if sig.side == Side.SELL:
            if not has_position:
                self._finish(sig, "REJECTED", "SELL ignored - no open long (strategy is long/flat, never shorts)",
                             "ignored - no open long position to sell (this strategy never shorts)")
                return
            closed = await self.order_manager.close_on_signal(symbol, sig.side, now, strategy_id=sig.strategy_id)
            if closed:
                qty = sum(t.position.qty for t in closed)
                net = sum(t.pnl_net for t in closed)
                self._finish(sig, "ACCEPTED", note or None,
                             f"SOLD {qty} @ {closed[0].exit_price:,.2f} - closed the long, net P&L {net:+,.2f}")
            else:
                self._finish(sig, "REJECTED", "exit order failed",
                             "exit order FAILED - the long is still open, retrying")
            return
        if not self._is_watched(symbol):
            self._record(sig, "REJECTED", "BUY ignored - symbol is not on the stock list")
            return
        if has_position:
            self._finish(sig, "REJECTED", "BUY ignored - already long", "ignored - already holding this stock")
            return
        if self.trade_start is not None and now.time() < self.trade_start:
            hhmm = self.trade_start.strftime("%H:%M")
            self._finish(sig, "REJECTED", f"no buys before {hhmm}", f"SKIPPED - trading starts at {hhmm}")
            return
        if self.square_off_time is not None and now.time() >= self.square_off_time:
            hhmm = self.square_off_time.strftime("%H:%M")
            self._finish(sig, "REJECTED", f"no new buys at/after the {hhmm} square-off",
                         f"SKIPPED - intraday only: every position is closed at {hhmm}, so no new buys after that")
            return
        open_count = len(self.order_manager.open_positions)
        if self.max_open_positions and open_count >= self.max_open_positions:
            self._finish(sig, "REJECTED", f"max open positions {open_count}/{self.max_open_positions}",
                         f"SKIPPED - {open_count}/{self.max_open_positions} positions already open; "
                         "a new MACD buy is taken only after a stock exits")
            return
        pct, why = sizing.choose_pct(ctx, price, self.default_position_pct, self.min_position_pct, self.max_position_pct)
        free_cash = self.capital + self.order_manager.realized_pnl() - self.order_manager.invested()
        plan = sizing.plan_size(price, self.capital, free_cash, open_count, self.max_open_positions, pct,
                                self.min_position_pct, self.max_position_pct)
        if plan.qty < 1:
            self._finish(sig, "REJECTED", f"BUY skipped - {plan.skip_reason}", f"SKIPPED - {plan.skip_reason}")
            return
        pos = await self.order_manager.open(sig, plan.qty, now)
        if pos:
            self._finish(sig, "ACCEPTED", note or None,
                         f"BOUGHT {pos.qty} @ {pos.entry_price:,.2f} - total {pos.qty * pos.entry_price:,.2f} "
                         f"({pos.qty * pos.entry_price / self.capital:.1%} of capital; target size {pct:.0%}: {why}; "
                         f"{open_count + 1}{f'/{self.max_open_positions}' if self.max_open_positions else ''} positions open)")
        else:
            self._finish(sig, "REJECTED", "entry order failed", "entry order FAILED - nothing bought")

    def _finish(self, sig: Signal, status: str, reject_reason: str | None, action: str) -> None:
        """Log the signal to the DB audit table AND tell Telegram what the system did with it."""
        self._record(sig, status, reject_reason)
        if self.recorder is not None:
            self.recorder.note_decision(sig.symbol, _bucket_start(sig.ts, sig.timeframe or self._live_timeframe()),
                                        sig.strategy_id, sig.side.value, sig.timeframe or "",
                                        sig.entry if sig.entry is not None else 0.0, status, reject_reason, action)
        if self.notify_signals:
            close = sig.entry if sig.entry is not None else 0.0
            live = self.signal_mode == "live"
            tg.notify(tg.format_signal(sig.symbol, sig.strategy_id, sig.side.value, sig.timeframe or "",
                                       sig.ts.strftime("%Y-%m-%d %H:%M:%S" if live else "%Y-%m-%d %H:%M"), close, sig.reason,
                                       action, self.order_manager.mode.value, live=live))

    def _record(self, sig: Signal, status: str, reject_reason: str | None) -> None:
        if self.signal_sink is not None:
            try:
                self.signal_sink(sig, status, reject_reason)
            except Exception:  # noqa: BLE001 - an audit-log failure must never block trading
                log.exception("could not record signal")

    # ------------------------------------------------------------------ carried-over signals
    def _save_deferred(self) -> None:
        if self._deferred_file is None:
            return
        rows = [{"symbol": sym, "date": d.isoformat(), "side": sig.side.value, "strategy_id": sig.strategy_id,
                 "strategy_version": sig.strategy_version, "ts": sig.ts.isoformat(), "entry": sig.entry,
                 "stop_loss": sig.stop_loss, "reason": sig.reason, "timeframe": sig.timeframe,
                 "sizing": vars(self._deferred_ctx[sym]) if sym in self._deferred_ctx else None}
                for sym, (sig, d) in self._deferred.items()]
        self._deferred_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._deferred_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        tmp.replace(self._deferred_file)

    def _load_deferred(self) -> None:
        if self._deferred_file is None or not self._deferred_file.exists():
            return
        for r in json.loads(self._deferred_file.read_text(encoding="utf-8")):
            sig = Signal(symbol=r["symbol"], side=Side(r["side"]), horizon=Horizon.INTRADAY, stop_loss=r["stop_loss"],
                         strategy_id=r["strategy_id"], strategy_version=r["strategy_version"],
                         ts=pd.Timestamp(r["ts"]), timeframe=r["timeframe"], entry=r["entry"], reason=r["reason"])
            self._deferred[r["symbol"]] = (sig, date.fromisoformat(r["date"]))
            if r.get("sizing"):
                self._deferred_ctx[r["symbol"]] = sizing.SizeContext(**r["sizing"])
        if self._deferred:
            log.info("loaded signals carried over from the previous close",
                     extra={"data": {"symbols": sorted(self._deferred)}})

    # ------------------------------------------------------------------ series
    def _build_df(self, key: tuple[str, str]) -> pd.DataFrame:
        history = self.history_by_symbol_tf.get(key)
        today_df = pd.DataFrame([{"ts": c.start, "open": c.open, "high": c.high, "low": c.low, "close": c.close,
                                  "volume": c.volume} for c in self._today_bars[key]], columns=BAR_COLUMNS)
        frames = [f[BAR_COLUMNS] for f in (history, today_df) if f is not None and not f.empty]
        if not frames:
            return pd.DataFrame(columns=BAR_COLUMNS)
        combined = pd.concat(frames, ignore_index=True)
        # history's ts is read from parquet (ns resolution); today's rows come from python datetimes (us) -
        # pd.concat silently falls back to dtype=object on the mismatch, which breaks any later .dt accessor
        # (e.g. session_vwap) with no obvious error at this line
        combined["ts"] = pd.to_datetime(combined["ts"], utc=True).dt.tz_convert("Asia/Kolkata")
        # a bar present in both sources keeps the broker's official bar (history is listed first)
        combined = combined.drop_duplicates("ts", keep="first").sort_values("ts")
        return combined.tail(self.history_bars).reset_index(drop=True)

    def _warn_on_gap(self, candle: Candle, df: pd.DataFrame) -> None:
        """A missing bar silently skews MACD. Normal neighbours: the previous bar of the same day, or the
        previous session's closing bar when this is the first bar of the day."""
        if len(df) < 2:
            return
        prev = df["ts"].iloc[-2]
        step = timedelta(minutes=minutes(candle.timeframe))
        same_day_ok = prev.date() == candle.start.date() and candle.start - prev == step
        new_day_ok = prev.date() < candle.start.date() and candle.start.time() == SESSION_OPEN
        if not (same_day_ok or new_day_ok):
            log.warning("bar gap before this candle - MACD may differ from the broker chart until history is refreshed",
                        extra={"data": {"symbol": candle.symbol, "bar": candle.start.isoformat(), "previous": str(prev)}})


def _f(x) -> float | None:
    return None if x is None or x != x else float(x)
