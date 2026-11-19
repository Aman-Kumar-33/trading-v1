"""Clocks.

Everything time-dependent asks a Clock for "now" instead of calling datetime.now() directly.
Live trading uses RealClock; replay/backtests use ReplayClock, which the replay adapter moves forward
tick by tick. That's what lets the same engine code run on historical days.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


class Clock:
    def now(self) -> datetime:
        raise NotImplementedError


class RealClock(Clock):
    def __init__(self, tz: ZoneInfo = IST):
        self.tz = tz

    def now(self) -> datetime:
        return datetime.now(self.tz)


class ReplayClock(Clock):
    def __init__(self, start: datetime):
        if start.tzinfo is None:
            raise ValueError("ReplayClock needs a timezone-aware start time")
        self._now = start

    def now(self) -> datetime:
        return self._now

    def set(self, ts: datetime) -> None:
        if ts < self._now:
            return  # never move backwards
        self._now = ts
