"""Timeframe helpers shared by the candle builder, cleaner and downloader."""
from __future__ import annotations

from datetime import timedelta

_MINUTES = {"1m": 1, "2m": 2, "3m": 3, "5m": 5, "10m": 10, "15m": 15, "30m": 30, "1h": 60, "60m": 60}

SESSION_MINUTES = 375  # 09:15 -> 15:30


def is_intraday(tf: str) -> bool:
    return tf in _MINUTES


def minutes(tf: str) -> int:
    """Length of an intraday timeframe in minutes. Raises for daily."""
    try:
        return _MINUTES[tf]
    except KeyError:
        raise ValueError(f"not an intraday timeframe: {tf}") from None


def delta(tf: str) -> timedelta:
    if tf == "1d":
        return timedelta(days=1)
    return timedelta(minutes=minutes(tf))


def bars_per_session(tf: str) -> int:
    """Expected number of bars in a full 09:15-15:30 session (the last bar may be shorter, e.g. 1h)."""
    m = minutes(tf)
    return -(-SESSION_MINUTES // m)  # ceil division


def pandas_freq(tf: str) -> str:
    if tf == "1d":
        return "1D"
    return f"{minutes(tf)}min"
