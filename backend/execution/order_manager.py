"""Order manager: turns a risk-approved (Signal, qty) into a tracked Position, manages it tick by tick
(stop-loss and target), and closes it into a ClosedTrade with the costs the exchange actually charges.

Same-day trading with carry-over (the user's rule): intraday LONGS are bought as CNC (delivery product),
so if they haven't hit stop/target by the 15:10 square-off they're simply held overnight and exit the
next day(s) on the original stop/target (computed from the buy price), an opposite signal, or after
`max_carry_days`. A same-day buy+sell on CNC is still charged as intraday by the exchange; only a
position actually held overnight pays delivery STT/stamp duty and the DP charge. Intraday SHORTS can't
be held overnight in the NSE cash segment, so they are MIS and always square off at 15:10 (ahead of
Angel One's own 15:15 RMS auto square-off).

Fills: entry and exit go through `broker.place_order()` and `broker.wait_for_fill()` - PaperBroker fills
instantly, Angel One is polled until the order completes. A failed entry opens nothing; a failed exit
leaves the position open and is retried after `exit_retry_seconds` - a position is never forgotten.
Strategy performance is NOT recorded here (backend/ensemble/shadow_book.py scores every signal).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, time, timedelta

from backend.brokers.base import BrokerAdapter
from backend.core.models import Horizon, Mode, OrderRequest, Side, Signal, Tick
from backend.execution import cost_model as cm
from backend.notify import telegram as tg
from backend.risk.risk_manager import RiskManager

log = logging.getLogger(__name__)

SESSION_OPEN, SESSION_CLOSE = time(9, 15), time(15, 30)


@dataclass
class Position:
    symbol: str
    side: Side
    horizon: Horizon
    qty: int
    entry_price: float
    stop_loss: float
    target: float | None
    strategy_id: str
    strategy_version: str
    regime: str | None
    opened_at: datetime
    product: str = "MIS"      # MIS (intraday, must square off) | CNC (delivery, may be held overnight)
    carried: bool = False     # an intraday long held past the square-off
    initial_risk: float = 0.0   # abs(entry - original stop) per share, fixed at open - R-multiples for
                                # trailing are measured against this, never the stop's own current (moved) value

    @property
    def notional(self) -> float:
        return self.qty * self.entry_price


@dataclass
class ClosedTrade:
    position: Position
    exit_price: float
    exit_reason: str   # STOP | TARGET | SQUARE_OFF | TIMEOUT | SIGNAL | MANUAL
    closed_at: datetime
    pnl_gross: float
    costs: float
    pnl_net: float
    r_multiple: float | None


def _in_session(now: datetime) -> bool:
    return now.weekday() < 5 and SESSION_OPEN <= now.time() < SESSION_CLOSE


BREAKEVEN_TRIGGER_R = 2   # backtested on 50 real symbols: breakeven@1R/trail@2R netted -20,469; delaying
TRAIL_START_R = BREAKEVEN_TRIGGER_R + 1   # to breakeven@2R/trail@3R netted -2,803 - 1R was cutting
                                          # trades during normal early consolidation before they could run


def _trail_stop(position: Position, ltp: float) -> bool:
    """Lock in profit instead of giving it all back to a reversal: at +BREAKEVEN_TRIGGER_R favorable, move
    the stop to breakeven; beyond +TRAIL_START_R, trail it BREAKEVEN_TRIGGER_R behind the current price
    (2R, not 1R - the backtest that picked these thresholds tested them together). Measured off
    `initial_risk` (fixed at open), never the stop's own already-moved value, so this can only ratchet the
    stop in one direction. Returns True if the stop actually moved (so the caller only persists state when
    something changed)."""
    if position.initial_risk <= 0:
        return False
    direction = 1 if position.side == Side.BUY else -1
    favorable_r = direction * (ltp - position.entry_price) / position.initial_risk
    if favorable_r < BREAKEVEN_TRIGGER_R:
        return False
    new_stop = (position.entry_price + direction * position.initial_risk * (favorable_r - BREAKEVEN_TRIGGER_R)
               if favorable_r >= TRAIL_START_R else position.entry_price)
    if (direction == 1 and new_stop > position.stop_loss) or (direction == -1 and new_stop < position.stop_loss):
        position.stop_loss = new_stop
        return True
    return False


def _lock_in_profit_stop(position: Position, ltp: float) -> bool:
    """Separate from `_trail_stop` (which is effectively inert here - these strategies set a deliberately
    wide 10x-ATR stop_loss, so price rarely moves the multi-R distance that function needs). Once a
    position is profitable enough to cover its own estimated round-trip cost, ratchet the stop to the
    exact price where, if hit, net P&L would be ~0 - so a reversal can take back the open gain but never
    turn a winning trade net-negative. Only ever moves the stop in the favorable direction."""
    direction = 1 if position.side == Side.BUY else -1
    pnl_gross = direction * (ltp - position.entry_price) * position.qty
    if pnl_gross <= 0 or position.qty <= 0:
        return False
    costs = cm.actual_trade_cost(position.entry_price, ltp, position.qty, position.side,
                                 position.horizon, position.carried).total
    if pnl_gross - costs <= 0:
        return False
    new_stop = position.entry_price + direction * (costs / position.qty)
    if (direction == 1 and new_stop > position.stop_loss) or (direction == -1 and new_stop < position.stop_loss):
        position.stop_loss = new_stop
        return True
    return False


class OrderManager:
    def __init__(self, broker: BrokerAdapter, risk_manager: RiskManager, mode: Mode = Mode.PAPER,
                 square_off_time: time = time(15, 10), max_holding_days: int = 90, carry_longs: bool = True,
                 max_carry_days: int = 3, journal=None, fill_timeout: float = 30.0, exit_retry_seconds: int = 10,
                 profit_lock_stop: bool = True, trail_stop: bool = True, stop_loss_pct: float | None = None):
        self.broker = broker
        self.risk_manager = risk_manager
        self.mode = mode
        self.square_off_time = square_off_time
        self.max_holding_days = max_holding_days
        self.carry_longs = carry_longs
        self.max_carry_days = max_carry_days
        self.journal = journal
        self.fill_timeout = fill_timeout
        self.exit_retry = timedelta(seconds=exit_retry_seconds)
        # Both move a position's stop on their own and exit it on a plain price tick - i.e. sells that are NOT
        # strategy signals. Off for the MACD-only system, where every exit must be a MACD cross.
        self.profit_lock_stop = profit_lock_stop
        self.trail_stop = trail_stop
        # a fixed stop for every new position: this share of the FILL price (None = the stop the signal carries)
        self.stop_loss_pct = float(stop_loss_pct) if stop_loss_pct else None
        self.open_positions: list[Position] = []
        self.closed_trades: list[ClosedTrade] = []
        self._exit_failed_at: dict[str, datetime] = {}

    # ------------------------------------------------------------------ queries / restore
    def has_position(self, symbol: str, strategy_id: str | None = None) -> bool:
        return any(p.symbol == symbol and (strategy_id is None or p.strategy_id == strategy_id)
                  for p in self.open_positions)

    def invested(self) -> float:
        """Money currently tied up in open positions (at entry price)."""
        return sum(p.entry_price * p.qty for p in self.open_positions)

    def realized_pnl(self) -> float:
        """Net P&L of the trades closed by this process - added to capital when working out free cash."""
        return sum(t.pnl_net for t in self.closed_trades)

    async def close_manual(self, symbol: str, now: datetime) -> list[ClosedTrade]:
        """A user-triggered exit from the dashboard - same order path as a stop/square-off close, just
        tagged distinctly so it's visible in the trades table that a human intervened. Closes EVERY open
        position on this symbol, not just one - the 3 strategies run in parallel and can each hold their
        own position on the same symbol at once, so "force exit this stock" must mean all of them, not
        whichever happened to be first in the list (that silently left others open before this fix)."""
        positions = [p for p in self.open_positions if p.symbol == symbol]
        closed = []
        for position in positions:
            trade = await self._close(position, "MANUAL", now)
            if trade:
                closed.append(trade)
        return closed

    def restore(self, rows: list[dict]) -> None:
        """Re-adopt positions saved by the journal (e.g. longs carried overnight) after a restart."""
        for d in rows:
            p = Position(**d)
            if p.initial_risk <= 0:
                # pre-trailing-stop state file (or any other legacy record missing the field) - backfill
                # from the stop currently on record, so a restart doesn't permanently exempt a carried
                # position from profit protection (see _trail_stop's initial_risk <= 0 guard)
                p.initial_risk = abs(p.entry_price - p.stop_loss)
            self.open_positions.append(p)
            self.risk_manager.on_position_opened(p.horizon, p.notional)
        if rows:
            log.info("restored open positions", extra={"data": {"count": len(rows),
                                                                "symbols": [d["symbol"] for d in rows]}})

    def _persist(self) -> None:
        if self.journal is not None:
            self.journal.save_positions(self.open_positions)

    def _product_for(self, signal: Signal) -> str:
        if signal.horizon == Horizon.LONG_TERM:
            return "CNC"
        return "CNC" if (signal.side == Side.BUY and self.carry_longs) else "MIS"

    # ------------------------------------------------------------------ entries
    async def open(self, signal: Signal, qty: int, now: datetime) -> Position | None:
        """Always a MARKET order - `signal.entry` is the strategy's reference price, not a limit."""
        product = self._product_for(signal)
        order = OrderRequest(symbol=signal.symbol, side=signal.side, qty=qty, order_type="MARKET",
                             product=product, tag=signal.strategy_id)
        try:
            order_id = await self.broker.place_order(order)
            fill_price, filled = await self.broker.wait_for_fill(order_id, self.fill_timeout)
        except Exception as exc:  # noqa: BLE001 - a failed entry must not crash the engine
            log.error("entry order failed - no position opened", extra={"data": {
                "symbol": signal.symbol, "strategy": signal.strategy_id, "error": repr(exc)}})
            return None
        stop_loss = signal.stop_loss
        if self.stop_loss_pct:
            stop_loss = fill_price * (1 - self.stop_loss_pct) if signal.side == Side.BUY else fill_price * (1 + self.stop_loss_pct)
        position = Position(symbol=signal.symbol, side=signal.side, horizon=signal.horizon, qty=filled,
                            entry_price=fill_price, stop_loss=stop_loss,
                            target=signal.targets[0] if signal.targets else None,
                            strategy_id=signal.strategy_id, strategy_version=signal.strategy_version,
                            regime=signal.regime, opened_at=now, product=product,
                            initial_risk=abs(fill_price - stop_loss))
        self.open_positions.append(position)
        self.risk_manager.on_position_opened(position.horizon, position.notional)
        if (signal.side == Side.BUY and position.stop_loss >= fill_price) or \
           (signal.side == Side.SELL and position.stop_loss <= fill_price):
            # the fill landed at/past the stop (a volatile next-bar gap, or a mis-sized stop) - the
            # position is already structurally invalid, not just unlucky; flatten it immediately instead
            # of leaving an "open" position with zero or negative real protection (see the audit: 14 of
            # 44 live trades had exactly this shape - zero or wrong-side stop distance)
            log.warning("stop already breached at fill - flattening immediately, no real position held",
                       extra={"data": {"symbol": signal.symbol, "strategy": signal.strategy_id,
                                       "fill": fill_price, "stop": position.stop_loss}})
            await self._close(position, "INVALID_STOP_AT_FILL", now)
            return None
        self._persist()
        log.info("position opened", extra={"data": {"symbol": position.symbol, "side": position.side.value,
                                                     "qty": filled, "entry": fill_price, "product": product,
                                                     "strategy": position.strategy_id}})
        tg.notify(tg.format_entry(position.symbol, position.strategy_id, position.side.value, filled, fill_price,
                                  self.mode.value, now.strftime("%Y-%m-%d %H:%M:%S")))
        return position

    # ------------------------------------------------------------------ exits
    async def on_tick(self, tick: Tick) -> list[ClosedTrade]:
        closed = []
        trailed = False
        for position in [p for p in self.open_positions if p.symbol == tick.symbol]:
            if self.trail_stop:
                trailed |= _trail_stop(position, tick.ltp)
            if self.profit_lock_stop and _lock_in_profit_stop(position, tick.ltp):
                trailed = True
            stop_hit = tick.ltp <= position.stop_loss if position.side == Side.BUY else tick.ltp >= position.stop_loss
            target_hit = (position.target is not None and
                         (tick.ltp >= position.target if position.side == Side.BUY else tick.ltp <= position.target))
            if stop_hit or target_hit:
                trade = await self._close(position, "STOP" if stop_hit else "TARGET", tick.ts)
                if trade:
                    closed.append(trade)
        if trailed and not closed:
            self._persist()
        return closed

    async def close_on_signal(self, symbol: str, new_side: Side, now: datetime,
                              strategy_id: str | None = None) -> list[ClosedTrade]:
        """An allowed strategy now signals the opposite direction - that's the exit signal."""
        closed = []
        for position in [p for p in self.open_positions if p.symbol == symbol and p.side != new_side
                         and (strategy_id is None or p.strategy_id == strategy_id)]:
            trade = await self._close(position, "SIGNAL", now)
            if trade:
                closed.append(trade)
        return closed

    async def on_time(self, now: datetime) -> list[ClosedTrade]:
        """Square off MIS positions at the cutoff, carry CNC intraday longs overnight, and time out carried
        longs after `max_carry_days` / long-term positions after `max_holding_days`. Only acts while the
        market is open - orders can't be placed outside the session."""
        if not _in_session(now):
            return []
        closed = []
        for position in list(self.open_positions):
            held_days = (now.date() - position.opened_at.date()).days
            if position.horizon == Horizon.INTRADAY:
                if position.carried or position.product == "CNC":
                    if held_days > self.max_carry_days:
                        trade = await self._close(position, "TIMEOUT", now)
                        closed += [trade] if trade else []
                    elif now.time() >= self.square_off_time and not position.carried and held_days == 0:
                        position.carried = True
                        self._persist()
                        log.info("intraday long carried overnight (exits on stop/target/signal next day)",
                                 extra={"data": {"symbol": position.symbol, "entry": position.entry_price,
                                                 "stop": position.stop_loss}})
                elif now.time() >= self.square_off_time or held_days > 0:
                    # held_days > 0: an MIS position that missed its square-off (process restart/outage
                    # during a prior session) - never meant to carry, so close it the instant trading
                    # resumes instead of letting it ride as a live position into a new day
                    trade = await self._close(position, "SQUARE_OFF", now)
                    closed += [trade] if trade else []
            elif held_days > self.max_holding_days:
                trade = await self._close(position, "TIMEOUT", now)
                closed += [trade] if trade else []
        return closed

    async def _close(self, position: Position, exit_reason: str, now: datetime) -> ClosedTrade | None:
        failed_at = self._exit_failed_at.get(position.symbol)
        if failed_at is not None and now - failed_at < self.exit_retry:
            return None
        exit_side = Side.SELL if position.side == Side.BUY else Side.BUY
        order = OrderRequest(symbol=position.symbol, side=exit_side, qty=position.qty, order_type="MARKET",
                             product=position.product, tag=position.strategy_id)
        try:
            order_id = await self.broker.place_order(order)
            exit_price, _ = await self.broker.wait_for_fill(order_id, self.fill_timeout)
        except Exception as exc:  # noqa: BLE001 - keep the position, retry later, shout about it
            self._exit_failed_at[position.symbol] = now
            log.critical("EXIT ORDER FAILED - position still open, will retry", extra={"data": {
                "symbol": position.symbol, "reason": exit_reason, "error": repr(exc)}})
            return None
        self._exit_failed_at.pop(position.symbol, None)

        direction = 1 if position.side == Side.BUY else -1
        pnl_gross = direction * (exit_price - position.entry_price) * position.qty
        overnight = now.date() != position.opened_at.date()
        costs = cm.actual_trade_cost(position.entry_price, exit_price, position.qty, position.side,
                                     position.horizon, overnight).total
        pnl_net = pnl_gross - costs
        # the ORIGINAL risk, not the current stop_loss - trailing moves the stop (even to exactly
        # breakeven), which would otherwise make a trade's R-multiple reflect how far it trailed instead
        # of how much was actually risked at entry (a real bug this surfaced: a trail-to-breakeven and a
        # target hit landing on the same tick produced a zero-distance "risk" and a None r_multiple)
        risk_per_share = position.initial_risk or abs(position.entry_price - position.stop_loss)
        r_multiple = (pnl_gross / position.qty) / risk_per_share if risk_per_share > 0 else None

        trade = ClosedTrade(position, exit_price, exit_reason, now, pnl_gross, costs, pnl_net, r_multiple)
        self.open_positions.remove(position)
        self.closed_trades.append(trade)
        self.risk_manager.on_position_closed(position.horizon, position.notional)
        self.risk_manager.on_trade_closed(pnl_net)
        if self.journal is not None:
            self.journal.record_trade(trade)
        self._persist()
        log.info("position closed", extra={"data": {"symbol": position.symbol, "exit_reason": exit_reason,
                                                     "pnl_net": round(pnl_net, 2), "overnight": overnight,
                                                     "strategy": position.strategy_id}})
        tg.notify(tg.format_exit(position.symbol, position.strategy_id, position.side.value, position.qty,
                                 exit_price, position.entry_price, pnl_gross, costs, pnl_net, exit_reason,
                                 self.mode.value, now.strftime("%Y-%m-%d %H:%M:%S")))
        return trade
