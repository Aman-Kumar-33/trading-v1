"""Replay adapter - plays a historical trading day through the live engine as if it were happening now.

Reads clean 1-minute bars from backtest_data/clean/1m/ and turns each bar into 4 synthetic ticks:
    :00 open  ->  :15 low/high  ->  :30 high/low  ->  :59 close
(low first for up-bars, high first for down-bars - the most likely path). Cumulative volume is split
across the ticks so rebuilt candles reproduce the bar volume exactly.

speed = 0     as fast as possible (tests, parity checks)
speed = 60    one market minute per real second
speed = 1     real time
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta
from typing import Awaitable, Callable

import pandas as pd

from backend.brokers.base import BrokerAdapter
from backend.core import events
from backend.core.clock import IST, ReplayClock
from backend.core.config import Settings
from backend.core.events import EventBus
from backend.core.models import Tick
from backend.data.universe import yf_symbol
from backend.storage import parquet_store

log = logging.getLogger(__name__)

TimeHook = Callable[[datetime], Awaitable[None]]


def bars_to_ticks(symbol: str, bars: pd.DataFrame, cum_start: int = 0) -> list[Tick]:
    """Synthetic tick path for 1-minute bars (columns ts, open, high, low, close, volume)."""
    ticks: list[Tick] = []
    cum = cum_start
    offsets = (timedelta(seconds=0), timedelta(seconds=15), timedelta(seconds=30), timedelta(seconds=59))
    for row in bars.itertuples(index=False):
        ts: datetime = row.ts.to_pydatetime() if hasattr(row.ts, "to_pydatetime") else row.ts
        o, h, l, c = float(row.open), float(row.high), float(row.low), float(row.close)
        path = (o, l, h, c) if c >= o else (o, h, l, c)
        v = int(row.volume)
        parts = (v // 4, v // 4, v // 4, v - 3 * (v // 4))
        for off, price, part in zip(offsets, path, parts):
            cum += part
            ticks.append(Tick(symbol=symbol, ts=ts + off, ltp=price, cum_volume=cum))
    return ticks


class ReplayAdapter(BrokerAdapter):
    name = "replay"

    def __init__(self, settings: Settings, bus: EventBus, clock: ReplayClock, trading_date: date,
                 interval: str = "1m", speed: float = 0.0):
        super().__init__(bus)
        self.settings = settings
        self.clock = clock
        self.trading_date = trading_date
        self.interval = interval
        self.speed = speed
        self._bars: dict[str, pd.DataFrame] = {}
        self.ticks_sent = 0

    async def connect(self) -> None:
        missing = []
        for sym in sorted(self.subscribed):
            df = self._load_day(sym)
            if df is None or df.empty:
                missing.append(sym)
                continue
            self._bars[sym] = df
        if missing:
            log.warning("no replay data for some symbols", extra={"data": {"date": self.trading_date.isoformat(),
                                                                          "symbols": missing}})

    async def disconnect(self) -> None:
        self._bars.clear()

    def _load_day(self, symbol: str) -> pd.DataFrame | None:
        path = self.settings.paths.clean / self.interval / f"{parquet_store.file_stem(yf_symbol(symbol))}.parquet"
        df = parquet_store.read(path)
        if df is None:
            return None
        return df.loc[df["ts"].dt.date == self.trading_date].reset_index(drop=True)

    def available_symbols(self) -> list[str]:
        return sorted(self._bars)

    def day_bars(self, symbol: str) -> pd.DataFrame | None:
        return self._bars.get(symbol)

    async def get_historical(self, symbol: str, interval: str, start: date, end: date) -> pd.DataFrame:
        df = parquet_store.read(self.settings.paths.clean / interval / f"{parquet_store.file_stem(yf_symbol(symbol))}.parquet")
        if df is None:
            return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
        d = df["ts"].dt.date
        # never leak the replay day's future into "history"
        end = min(end, self.trading_date - timedelta(days=1)) if interval == "1d" else end
        return df.loc[(d >= start) & (d <= end)].reset_index(drop=True)

    async def run(self, on_time: list[TimeHook] | None = None, finish_at: datetime | None = None) -> int:
        """Stream the day's ticks. `on_time` hooks are awaited whenever the clock moves (before the tick
        is published), e.g. SessionManager.step and MarketDataService.on_clock."""
        hooks = on_time or []
        ticks: list[Tick] = []
        for sym, bars in self._bars.items():
            ticks.extend(bars_to_ticks(sym, bars))
        ticks.sort(key=lambda t: (t.ts, t.symbol))
        prev_ts: datetime | None = None
        for tick in ticks:
            if self.speed > 0 and prev_ts is not None and tick.ts > prev_ts:
                await asyncio.sleep((tick.ts - prev_ts).total_seconds() / self.speed)
            if prev_ts is None or tick.ts != prev_ts:
                self.clock.set(tick.ts)
                for hook in hooks:
                    await hook(tick.ts)
                prev_ts = tick.ts
            await self.bus.publish(events.TICK, tick)
            self.ticks_sent += 1
        if finish_at is not None:
            self.clock.set(finish_at)
            for hook in hooks:
                await hook(finish_at)
        return self.ticks_sent


def replay_start_time(trading_date: date) -> datetime:
    return datetime.combine(trading_date, datetime.min.time(), tzinfo=IST).replace(hour=8, minute=55)
