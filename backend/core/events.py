"""In-process publish/subscribe event bus (replaces Redis pub/sub - no server needed).

Components talk through topics instead of calling each other directly:
    broker adapter --TICK--> candle builder --CANDLE_CLOSED--> strategies / storage / dashboard

Handlers may be plain functions or async coroutines. A failing handler is logged and does not stop
other handlers from receiving the event.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
from collections import defaultdict
from typing import Any, Awaitable, Callable

log = logging.getLogger(__name__)

Handler = Callable[[Any], Any] | Callable[[Any], Awaitable[Any]]

# Topic names
TICK = "tick"
CANDLE_CLOSED = "candle.closed"
CANDLE_UPDATED = "candle.updated"     # forming candle changed (for live charts); high frequency
SESSION = "session"                   # session state transitions (pre-open, open, square-off, close, eod freeze)


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[str, list[Handler]] = defaultdict(list)
        self.published: dict[str, int] = defaultdict(int)

    def subscribe(self, topic: str, handler: Handler) -> None:
        self._subs[topic].append(handler)

    def unsubscribe(self, topic: str, handler: Handler) -> None:
        if handler in self._subs.get(topic, []):
            self._subs[topic].remove(handler)

    def has_subscribers(self, topic: str) -> bool:
        return bool(self._subs.get(topic))

    async def publish(self, topic: str, payload: Any) -> None:
        self.published[topic] += 1
        for handler in list(self._subs.get(topic, [])):
            try:
                result = handler(payload)
                if inspect.isawaitable(result):
                    await result
            except Exception:  # noqa: BLE001 - one bad subscriber must not break the pipeline
                log.exception("event handler failed", extra={"data": {"topic": topic, "handler": repr(handler)}})

    def publish_sync(self, topic: str, payload: Any) -> None:
        """Publish from synchronous code. Async handlers are scheduled on the running loop if any,
        otherwise run to completion immediately."""
        self.published[topic] += 1
        for handler in list(self._subs.get(topic, [])):
            try:
                result = handler(payload)
                if inspect.isawaitable(result):
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(result)
                    except RuntimeError:
                        asyncio.run(result)
            except Exception:  # noqa: BLE001
                log.exception("event handler failed", extra={"data": {"topic": topic, "handler": repr(handler)}})
