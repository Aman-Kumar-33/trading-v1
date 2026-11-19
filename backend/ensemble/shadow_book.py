"""Live counterpart of full_system_backtest's shadow tracking: every signal every strategy emits gets a
virtual position (filled at the next tick, like a market order), managed tick by tick with the same
stop/target/square-off/timeout rules as a real one, and scored in net R after Angel One's real costs
when it closes. That score is what the PerformanceTracker learns from - so a muted strategy keeps being
measured and can earn its weight back, RESEARCH-status strategies get a free forward test while they're
not allowed to trade, and confluence doesn't credit only the strategy the allocator happened to pick.

Virtual positions never touch the broker, the risk manager's limits, or the real order book.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, time

from backend.core.models import Horizon, Side, Signal, Tick
from backend.ensemble.performance_tracker import PerformanceTracker
from backend.execution import cost_model as cm
from backend.risk.risk_manager import RiskManager, not_executable_reason

log = logging.getLogger(__name__)


@dataclass
class _ShadowPosition:
    signal: Signal
    opened_at: datetime
    atr: float | None
    qty: int = 0
    entry_price: float | None = None   # None until the next tick for the symbol fills it


class ShadowBook:
    def __init__(self, tracker: PerformanceTracker, risk_manager: RiskManager,
                 square_off_time: time = time(15, 10), max_holding_days: int = 90, carry_longs: bool = True,
                 max_carry_days: int = 3):
        self.tracker = tracker
        self.risk_manager = risk_manager   # only for sizing (nominal_qty) and the executability rule
        self.square_off_time = square_off_time
        self.max_holding_days = max_holding_days
        self.carry_longs = carry_longs        # same carry-over rule as OrderManager
        self.max_carry_days = max_carry_days
        self.positions: dict[tuple[str, str], _ShadowPosition] = {}
        self._last_price: dict[str, float] = {}
        self._last_seen: datetime | None = None
        self.recorded = 0

    def open(self, signal: Signal, now: datetime, atr: float | None = None) -> None:
        key = (signal.strategy_id, signal.symbol)
        if key in self.positions or not_executable_reason(signal, self.risk_manager.limits):
            return
        self.positions[key] = _ShadowPosition(signal=signal, opened_at=now, atr=atr)

    def on_tick(self, tick: Tick) -> None:
        self._last_price[tick.symbol] = tick.ltp
        self._last_seen = tick.ts
        for key, pos in list(self.positions.items()):
            if pos.signal.symbol != tick.symbol:
                continue
            sig = pos.signal
            if pos.entry_price is None:
                stale = sig.horizon == Horizon.INTRADAY and tick.ts.date() != pos.opened_at.date()
                breached = (sig.side == Side.BUY and sig.stop_loss >= tick.ltp) or \
                           (sig.side == Side.SELL and sig.stop_loss <= tick.ltp)
                qty = 0 if (stale or breached) else self.risk_manager.nominal_qty(sig, tick.ltp)
                if qty <= 0:
                    del self.positions[key]
                    continue
                pos.entry_price, pos.qty = tick.ltp, qty
                continue
            stop_hit = tick.ltp <= sig.stop_loss if sig.side == Side.BUY else tick.ltp >= sig.stop_loss
            target = sig.targets[0] if sig.targets else None
            target_hit = target is not None and (tick.ltp >= target if sig.side == Side.BUY else tick.ltp <= target)
            if stop_hit or target_hit:
                self._close(key, tick.ltp)

    def on_time(self, now: datetime) -> None:
        self._last_seen = now if self._last_seen is None or now > self._last_seen else self._last_seen
        for key, pos in list(self.positions.items()):
            intraday = pos.signal.horizon == Horizon.INTRADAY
            held_days = (now.date() - pos.opened_at.date()).days
            carried_long = intraday and self.carry_longs and pos.signal.side == Side.BUY
            if carried_long:
                if pos.entry_price is None and now.time() >= self.square_off_time:
                    del self.positions[key]      # never filled today - nothing to carry
                elif held_days > self.max_carry_days:
                    self._close(key, self._last_price.get(pos.signal.symbol))
            elif intraday and now.time() >= self.square_off_time:
                self._close(key, self._last_price.get(pos.signal.symbol))
            elif not intraday and held_days > self.max_holding_days:
                self._close(key, self._last_price.get(pos.signal.symbol))

    def _close(self, key: tuple[str, str], exit_price: float | None) -> None:
        pos = self.positions.pop(key)
        if pos.entry_price is None or exit_price is None:
            return   # never filled - nothing to learn from
        sig = pos.signal
        direction = 1 if sig.side == Side.BUY else -1
        pnl_gross = direction * (exit_price - pos.entry_price) * pos.qty
        overnight = self._last_seen is not None and self._last_seen.date() != pos.opened_at.date()
        costs = cm.actual_trade_cost(pos.entry_price, exit_price, pos.qty, sig.side, sig.horizon, overnight).total
        if pos.atr:
            costs += cm.slippage(pos.entry_price, pos.atr, pos.qty)
        risk_amount = pos.qty * abs(pos.entry_price - sig.stop_loss)
        # a stop pinned too close to entry makes R explode (e.g. 47R on one bad signal) and corrupts the
        # tracker's rolling mean for that strategy+regime for the next `window` trades - skip it instead.
        if risk_amount <= 0 or abs(pos.entry_price - sig.stop_loss) < pos.entry_price * 0.001:
            return
        r_net = (pnl_gross - costs) / risk_amount
        self.tracker.record(sig.strategy_id, sig.regime, r_net)
        self.recorded += 1
        log.info("shadow trade scored", extra={"data": {"strategy": sig.strategy_id, "symbol": sig.symbol,
                                                         "regime": sig.regime, "r_net": round(r_net, 3)}})
