"""Paper broker: simulates fills at the current tick price plus a simple ATR-based slippage estimate
(backend.execution.cost_model.slippage - the same estimate the backtester uses), so OrderManager can
place entry and exit orders through the exact same BrokerAdapter interface a real broker uses, without
ever touching a real account. Transaction costs (brokerage/STT/GST/stamp duty) are NOT applied here -
those depend on the full round trip (entry + exit together), which only OrderManager knows about; see
backend.execution.cost_model.round_trip_cost, applied when a position closes.

Deliberately simple for Phase 4 (paper trading only): fills are instant and synchronous (`place_order`
returns a price immediately via `filled_orders`), which is NOT how a real broker works (Phase 8's job -
order acknowledgement, partial fills, rejections - is a live-trading concern, not a paper one).
"""
from __future__ import annotations

import itertools
from typing import Any

from backend.brokers.base import BrokerAdapter
from backend.core import events
from backend.core.events import EventBus
from backend.core.models import OrderRequest, Side
from backend.execution import cost_model as cm


class PaperBroker(BrokerAdapter):
    name = "paper"
    supports_data = False
    supports_trading = True

    def __init__(self, bus: EventBus, starting_cash: float, atr_by_symbol: dict[str, float] | None = None):
        super().__init__(bus)
        self.cash = starting_cash
        self.atr_by_symbol = atr_by_symbol or {}
        self._last_price: dict[str, float] = {}
        self._order_seq = itertools.count(1)
        self.filled_orders: dict[str, dict[str, Any]] = {}

    async def connect(self) -> None:
        self.bus.subscribe(events.TICK, self._on_tick)

    async def disconnect(self) -> None:
        self.bus.unsubscribe(events.TICK, self._on_tick)

    def _on_tick(self, tick) -> None:
        self._last_price[tick.symbol] = tick.ltp

    async def get_historical(self, symbol: str, interval: str, start, end):
        raise NotImplementedError("PaperBroker is trading-only - pair it with a data-capable adapter for history")

    def set_price(self, symbol: str, price: float) -> None:
        """Lets a caller (e.g. a replay-driven test) set a symbol's price directly without needing a
        real Tick/event-bus round trip - the same thing `_on_tick` does, exposed for convenience."""
        self._last_price[symbol] = price

    def _fill_price(self, symbol: str, side: Side, qty: int) -> float:
        ltp = self._last_price.get(symbol)
        if ltp is None:
            raise RuntimeError(f"no known price for {symbol} yet - no tick has arrived")
        atr = self.atr_by_symbol.get(symbol, 0.0)
        slip_per_share = (cm.slippage(ltp, atr, qty) / qty) if qty else 0.0
        return ltp + slip_per_share if side == Side.BUY else ltp - slip_per_share

    async def place_order(self, order: OrderRequest) -> str:
        price = (order.price if order.order_type == "LIMIT" and order.price is not None
                else self._fill_price(order.symbol, order.side, order.qty))
        order_id = f"PAPER-{next(self._order_seq)}"
        self.filled_orders[order_id] = {"order": order, "fill_price": price}
        return order_id

    async def wait_for_fill(self, order_id: str, timeout: float = 30.0) -> tuple[float, int]:
        fill = self.filled_orders[order_id]
        return fill["fill_price"], fill["order"].qty

    async def modify_order(self, order_id: str, **changes: Any) -> None:
        raise NotImplementedError("paper fills are instant - there is nothing to modify after placing")

    async def cancel_order(self, order_id: str) -> None:
        raise NotImplementedError("paper fills are instant - there is nothing to cancel after placing")

    async def positions(self) -> list[dict[str, Any]]:
        return []

    async def funds(self) -> dict[str, Any]:
        return {"cash": self.cash}
