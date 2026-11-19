"""Rate-limit protection for Yahoo Finance (and any other free data source).

Three layers:
  1. TokenBucket     - caps requests per minute globally, so we never hammer the server.
  2. RateLimitGuard  - when the server still says "too many requests": exponential backoff with jitter,
                       and a circuit breaker that pauses everything for a long time after repeated hits.
  3. ProcessLock     - only one downloader process can run at a time (two would double the request rate).

Sleep and clock functions are injectable so tests run instantly.
"""
from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

_RATE_LIMIT_MARKERS = ("ratelimit", "rate limit", "too many requests", "429",
                      "exceeding access rate", "access rate", "access denied because")


def is_rate_limit_error(err: object) -> bool:
    """Shared across data sources (yfinance, Angel One SmartAPI, ...): each has its own wording for
    "you're going too fast", so this list grows as we add sources rather than being source-specific."""
    text = (repr(err) if not isinstance(err, str) else err).lower().replace("_", " ")
    compact = text.replace(" ", "")
    return any(m.replace(" ", "") in compact for m in _RATE_LIMIT_MARKERS)


class TokenBucket:
    """Thread-safe token bucket. `acquire()` blocks until a request is allowed."""

    def __init__(self, requests_per_minute: float, burst: int = 1,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be > 0")
        self.rate = requests_per_minute / 60.0      # tokens per second
        self.capacity = max(1, int(burst))
        self.tokens = float(self.capacity)
        self._clock = clock
        self._sleep = sleep
        self._last = clock()
        self._lock = threading.Lock()
        self.total_wait = 0.0

    def _refill(self) -> None:
        now = self._clock()
        self.tokens = min(self.capacity, self.tokens + (now - self._last) * self.rate)
        self._last = now

    def acquire(self, n: int = 1) -> float:
        """Take `n` tokens (one per HTTP request about to be made), sleeping as needed.
        Returns total seconds waited."""
        return sum(self._acquire_one() for _ in range(max(1, n)))

    def _acquire_one(self) -> float:
        waited = 0.0
        while True:
            with self._lock:
                self._refill()
                if self.tokens >= 1:
                    self.tokens -= 1
                    self.total_wait += waited
                    return waited
                need = (1 - self.tokens) / self.rate
            self._sleep(need)
            waited += need


@dataclass
class RateLimitGuard:
    backoff_base: float = 30.0
    backoff_max: float = 900.0
    breaker_threshold: int = 3
    breaker_pause: float = 1800.0
    sleep: Callable[[float], None] = time.sleep
    rng: random.Random = field(default_factory=random.Random)

    consecutive_hits: int = 0
    total_hits: int = 0
    breaker_trips: int = 0

    def on_success(self) -> None:
        self.consecutive_hits = 0

    def on_rate_limit(self, attempt: int) -> float:
        """Call after a rate-limit response. Sleeps and returns the seconds slept."""
        self.consecutive_hits += 1
        self.total_hits += 1
        if self.consecutive_hits >= self.breaker_threshold:
            self.breaker_trips += 1
            self.consecutive_hits = 0
            log.warning("circuit breaker tripped - pausing downloader",
                        extra={"data": {"pause_s": self.breaker_pause, "trips": self.breaker_trips}})
            self.sleep(self.breaker_pause)
            return self.breaker_pause
        wait = self.backoff_seconds(attempt)
        log.warning("rate limited - backing off", extra={"data": {"attempt": attempt, "wait_s": round(wait, 1)}})
        self.sleep(wait)
        return wait

    def on_network_error(self, attempt: int) -> float:
        wait = min(self.backoff_max, 5.0 * (2 ** attempt)) + self.rng.uniform(0, 2)
        self.sleep(wait)
        return wait

    def backoff_seconds(self, attempt: int) -> float:
        base = min(self.backoff_max, self.backoff_base * (2 ** attempt))
        return base + self.rng.uniform(0, base * 0.25)   # jitter so retries don't line up


class LockHeld(RuntimeError):
    pass


class ProcessLock:
    """Exclusive lock file. A lock older than `stale_hours` is assumed abandoned (crashed process)."""

    def __init__(self, path: Path, stale_hours: float = 12.0):
        self.path = Path(path)
        self.stale_seconds = stale_hours * 3600
        self._held = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                age = time.time() - self.path.stat().st_mtime
                if age > self.stale_seconds:
                    log.warning("removing stale lock", extra={"data": {"path": str(self.path), "age_h": round(age / 3600, 1)}})
                    self.path.unlink(missing_ok=True)
                    continue
                raise LockHeld(f"another downloader is running (lock file {self.path}, {self._describe()}). "
                               f"If it crashed, delete the lock file or wait {self.stale_seconds / 3600:.0f}h.") from None
            with os.fdopen(fd, "w") as fh:
                json.dump({"pid": os.getpid(), "started": time.time()}, fh)
            self._held = True
            return
        raise LockHeld(f"could not acquire {self.path}")

    def _describe(self) -> str:
        try:
            return self.path.read_text()
        except OSError:
            return "unreadable"

    def release(self) -> None:
        if self._held:
            self.path.unlink(missing_ok=True)
            self._held = False

    def __enter__(self) -> "ProcessLock":
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()
