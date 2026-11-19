"""Angel One live market data: SmartWebSocketV2 in QUOTE mode (last price + cumulative day volume, which
the candle builder needs), turned into Tick events on the in-process event bus.

The SDK's `connect()` blocks forever, so it runs in a daemon thread; each message is handed to the
asyncio loop with `run_coroutine_threadsafe` so every downstream handler (candle builder, strategy runner,
order manager) runs on the loop thread as usual. Prices arrive in paise and timestamps in epoch
milliseconds - both converted here. `last_tick_at` lets the runner's heartbeat notice a silent feed.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime
from typing import Any, Callable

from backend.core import events
from backend.core.clock import IST
from backend.core.events import EventBus
from backend.core.models import Tick

log = logging.getLogger(__name__)

QUOTE_MODE = 2
NSE_CM = 1


def parse_quote(msg: dict[str, Any], token_to_symbol: dict[str, str]) -> Tick | None:
    symbol = token_to_symbol.get(str(msg.get("token")))
    ltp = msg.get("last_traded_price")
    ts_ms = msg.get("exchange_timestamp")
    if symbol is None or not ltp or not ts_ms:
        return None
    return Tick(symbol=symbol, ts=datetime.fromtimestamp(ts_ms / 1000, IST), ltp=ltp / 100.0,
                cum_volume=msg.get("volume_trade_for_the_day"))


class AngelTickFeed:
    def __init__(self, bus: EventBus, loop: asyncio.AbstractEventLoop, auth_token: str, api_key: str,
                 client_code: str, feed_token: str, token_to_symbol: dict[str, str],
                 ws_factory: Callable[..., Any] | None = None):
        self.bus = bus
        self.loop = loop
        self.token_to_symbol = token_to_symbol
        self.last_tick_at: datetime | None = None
        self.ticks = 0
        self._ws_args = (auth_token, api_key, client_code, feed_token)
        self._ws_factory = ws_factory
        self._ws = None
        self._thread: threading.Thread | None = None

    def _make_ws(self):
        if self._ws_factory is not None:
            return self._ws_factory(*self._ws_args)
        from SmartApi.smartWebSocketV2 import SmartWebSocketV2
        return SmartWebSocketV2(*self._ws_args, max_retry_attempt=5)

    def start(self) -> None:
        ws = self._make_ws()
        tokens = [{"exchangeType": NSE_CM, "tokens": list(self.token_to_symbol)}]
        # an empty list (the 5 minute system started before the stocks it follows are known) subscribes later
        ws.on_open = lambda _wsapp: ws.subscribe("algo-feed", QUOTE_MODE, tokens) if self.token_to_symbol else None
        ws.on_data = lambda _wsapp, msg: self.on_message(msg)
        ws.on_error = lambda *a: log.error("angel feed error", extra={"data": {"args": repr(a)[:300]}})
        ws.on_close = lambda *a: log.warning("angel feed closed")
        self._ws = ws
        self._thread = threading.Thread(target=ws.connect, name="angel-feed", daemon=True)
        self._thread.start()
        log.info("angel feed started", extra={"data": {"symbols": len(self.token_to_symbol)}})

    def subscribe_more(self, new_tokens: dict[str, str]) -> None:
        """Start receiving ticks for more symbols on the already-open socket ({token: symbol}). They are also
        added to `token_to_symbol`, so a reconnect re-subscribes to them too."""
        fresh = [t for t in new_tokens if t not in self.token_to_symbol]
        self.token_to_symbol.update(new_tokens)
        if fresh and self._ws is not None:
            self._ws.subscribe("algo-feed", QUOTE_MODE, [{"exchangeType": NSE_CM, "tokens": fresh}])
            log.info("angel feed: subscribed to more symbols", extra={"data": {"count": len(fresh)}})

    def on_message(self, msg: dict[str, Any]) -> None:
        """Runs on the websocket thread - never touch the bus directly from here."""
        tick = parse_quote(msg, self.token_to_symbol)
        if tick is None:
            return
        self.last_tick_at = tick.ts
        self.ticks += 1
        asyncio.run_coroutine_threadsafe(self.bus.publish(events.TICK, tick), self.loop)

    def stop(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close_connection()
            except Exception:  # noqa: BLE001 - shutting down anyway
                log.exception("error closing angel feed")
