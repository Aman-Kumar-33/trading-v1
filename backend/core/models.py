"""Core data types shared by every component."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


@dataclass(slots=True)
class Tick:
    """One market data update from a broker (or the replay adapter).

    ts            exchange timestamp, timezone-aware (IST)
    ltp           last traded price
    cum_volume    cumulative traded volume for the day, if the feed provides it (most broker feeds do)
    last_qty      quantity of the last trade, used when cum_volume is not available
    """
    symbol: str
    ts: datetime
    ltp: float
    cum_volume: int | None = None
    last_qty: int | None = None


@dataclass(slots=True)
class Candle:
    symbol: str
    timeframe: str          # "1m", "5m", "15m", "1h", "1d"
    start: datetime         # bucket start (inclusive), IST
    end: datetime           # bucket end (exclusive), IST
    open: float
    high: float
    low: float
    close: float
    volume: int = 0
    tick_count: int = 0
    closed: bool = False

    def update(self, price: float, volume: int) -> None:
        if price > self.high:
            self.high = price
        if price < self.low:
            self.low = price
        self.close = price
        self.volume += volume
        self.tick_count += 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SessionState(str, Enum):
    CLOSED = "CLOSED"            # non-trading day, or before pre-open / after close
    PRE_OPEN = "PRE_OPEN"        # 09:00-09:15 call auction
    OPEN = "OPEN"                # 09:15-15:15 continuous trading, entries allowed
    SQUARE_OFF = "SQUARE_OFF"    # 15:15-15:30 no new intraday entries, intraday positions closed
    POST_CLOSE = "POST_CLOSE"    # after 15:30 on a trading day, until EOD freeze
    EOD_FROZEN = "EOD_FROZEN"    # day finished; dashboard shows static end-of-day view


@dataclass(slots=True)
class SessionEvent:
    state: SessionState
    previous: SessionState | None
    ts: datetime
    trading_date: str | None = None
    info: dict[str, Any] = field(default_factory=dict)


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class Horizon(str, Enum):
    INTRADAY = "INTRADAY"
    LONG_TERM = "LONG_TERM"


class Mode(str, Enum):
    BACKTEST = "BACKTEST"
    PAPER = "PAPER"
    LIVE = "LIVE"


@dataclass(slots=True)
class OrderRequest:
    symbol: str
    side: Side
    qty: int
    order_type: str = "MARKET"      # MARKET | LIMIT | SL | SL-M
    product: str = "MIS"            # MIS (intraday) | CNC (delivery)
    price: float | None = None
    trigger_price: float | None = None
    tag: str | None = None          # strategy / algo id tag (needed for SEBI algo identification)


@dataclass(slots=True)
class Signal:
    """What a strategy emits. Mirrors the `signals` table (backend/storage/db.py) column for column.

    Every signal MUST carry a stop_loss - backend.strategies.base.Strategy.run() rejects (and logs)
    any signal a strategy tries to emit without one, before it ever reaches the risk manager.
    """
    symbol: str
    side: Side
    horizon: Horizon
    stop_loss: float
    strategy_id: str
    strategy_version: str
    ts: datetime | None = None
    timeframe: str | None = None
    entry: float | None = None      # None = enter at market on the next bar
    targets: list[float] = field(default_factory=list)
    confidence: float | None = None   # 0-100, typically the context_score.py composite
    reason: str = ""
    patterns: list[str] = field(default_factory=list)
    regime: str | None = None
    status: str = "NEW"             # NEW | ACCEPTED | REJECTED | EXPIRED
    reject_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["side"] = self.side.value
        d["horizon"] = self.horizon.value
        if self.ts is not None:
            d["ts"] = self.ts.isoformat()
        return d
