"""Broker adapter interface.

Every data/execution source implements this: ReplayAdapter (historical days replayed as live) now,
Angel One SmartAPI and Zerodha Kite once an account is available, and PaperBroker in Phase 4.
The rest of the system only talks to this interface, so switching brokers means writing one adapter.

Ticks are delivered by publishing `Tick` objects on the event bus (topic events.TICK).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Any

import pandas as pd

from backend.core.events import EventBus
from backend.core.models import OrderRequest


class BrokerAdapter(ABC):
    name: str = "base"
    supports_data: bool = True
    supports_trading: bool = False

    def __init__(self, bus: EventBus):
        self.bus = bus
        self.subscribed: set[str] = set()

    # ---------------------------------------------------------------- market data
    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def disconnect(self) -> None: ...

    async def subscribe(self, symbols: list[str]) -> None:
        self.subscribed.update(symbols)

    async def unsubscribe(self, symbols: list[str]) -> None:
        self.subscribed.difference_update(symbols)

    @abstractmethod
    async def get_historical(self, symbol: str, interval: str, start: date, end: date) -> pd.DataFrame:
        """OHLCV bars with columns ts, open, high, low, close, volume (ts timezone-aware IST)."""

    # ---------------------------------------------------------------- trading (Phase 4+)
    async def place_order(self, order: OrderRequest) -> str:
        raise NotImplementedError(f"{self.name} adapter is data-only")

    async def wait_for_fill(self, order_id: str, timeout: float = 30.0) -> tuple[float, int]:
        """(average fill price, filled quantity) once the order is complete. Raises if it's rejected,
        cancelled, or not filled within `timeout` seconds - callers must not assume a fill."""
        raise NotImplementedError(f"{self.name} adapter is data-only")

    async def modify_order(self, order_id: str, **changes: Any) -> None:
        raise NotImplementedError(f"{self.name} adapter is data-only")

    async def cancel_order(self, order_id: str) -> None:
        raise NotImplementedError(f"{self.name} adapter is data-only")

    async def positions(self) -> list[dict[str, Any]]:
        raise NotImplementedError(f"{self.name} adapter is data-only")

    async def funds(self) -> dict[str, Any]:
        raise NotImplementedError(f"{self.name} adapter is data-only")
