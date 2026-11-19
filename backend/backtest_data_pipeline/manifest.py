"""Download manifest - remembers what has been fetched so nothing is downloaded twice.

Stored as SQLite at backtest_data/manifest/manifest.db:
  fetch_state  one row per (symbol, interval): first/last bar stored, row count, status, last error
  fetch_log    one row per download attempt (audit trail)

If the downloader crashes or gets blocked, the next run reads fetch_state and continues from there.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from backend.storage.db import connect

_SCHEMA = """
CREATE TABLE IF NOT EXISTS fetch_state (
    symbol            TEXT NOT NULL,     -- Yahoo symbol, e.g. RELIANCE.NS
    interval          TEXT NOT NULL,
    first_bar         TEXT,              -- ISO timestamp of the oldest stored bar
    last_bar          TEXT,              -- ISO timestamp of the newest stored bar
    rows              INTEGER DEFAULT 0,
    status            TEXT,              -- ok | no_data | error | rate_limited
    error             TEXT,
    attempts          INTEGER DEFAULT 0,
    last_attempt_utc  TEXT,
    last_success_utc  TEXT,
    PRIMARY KEY (symbol, interval)
);
CREATE TABLE IF NOT EXISTS fetch_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc      TEXT NOT NULL,
    symbols     TEXT NOT NULL,
    interval    TEXT NOT NULL,
    start       TEXT,
    end         TEXT,
    rows        INTEGER,
    status      TEXT,
    error       TEXT,
    duration_s  REAL
);
"""


@dataclass
class FetchState:
    symbol: str
    interval: str
    first_bar: str | None
    last_bar: str | None
    rows: int
    status: str | None
    error: str | None
    attempts: int
    last_attempt_utc: str | None
    last_success_utc: str | None


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Manifest:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.conn: sqlite3.Connection = connect(self.path)
        self.conn.executescript(_SCHEMA)

    def get(self, symbol: str, interval: str) -> FetchState | None:
        row = self.conn.execute("SELECT * FROM fetch_state WHERE symbol=? AND interval=?", (symbol, interval)).fetchone()
        return FetchState(**dict(row)) if row else None

    def all(self, interval: str | None = None) -> list[FetchState]:
        if interval:
            rows = self.conn.execute("SELECT * FROM fetch_state WHERE interval=? ORDER BY symbol", (interval,))
        else:
            rows = self.conn.execute("SELECT * FROM fetch_state ORDER BY interval, symbol")
        return [FetchState(**dict(r)) for r in rows]

    def record_success(self, symbol: str, interval: str, first_bar: str, last_bar: str, rows: int,
                       note: str | None = None, now: str | None = None) -> None:
        now = now or _utcnow()
        self.conn.execute(
            """INSERT INTO fetch_state (symbol, interval, first_bar, last_bar, rows, status, error, attempts,
                                        last_attempt_utc, last_success_utc)
               VALUES (?, ?, ?, ?, ?, 'ok', ?, 1, ?, ?)
               ON CONFLICT(symbol, interval) DO UPDATE SET
                   first_bar=excluded.first_bar, last_bar=excluded.last_bar, rows=excluded.rows,
                   status='ok', error=excluded.error, attempts=fetch_state.attempts + 1,
                   last_attempt_utc=excluded.last_attempt_utc, last_success_utc=excluded.last_success_utc""",
            (symbol, interval, first_bar, last_bar, rows, note, now, now),
        )

    def record_failure(self, symbol: str, interval: str, status: str, error: str, now: str | None = None) -> None:
        now = now or _utcnow()
        self.conn.execute(
            """INSERT INTO fetch_state (symbol, interval, status, error, attempts, last_attempt_utc)
               VALUES (?, ?, ?, ?, 1, ?)
               ON CONFLICT(symbol, interval) DO UPDATE SET
                   status=excluded.status, error=excluded.error, attempts=fetch_state.attempts + 1,
                   last_attempt_utc=excluded.last_attempt_utc""",
            (symbol, interval, status, error[:1000], now),
        )

    def log_attempt(self, symbols: list[str], interval: str, start: str | None, end: str | None,
                    rows: int, status: str, error: str | None, duration_s: float, now: str | None = None) -> None:
        self.conn.execute(
            "INSERT INTO fetch_log (ts_utc, symbols, interval, start, end, rows, status, error, duration_s) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (now or _utcnow(), ",".join(symbols), interval, start, end, rows, status, (error or "")[:2000],
             round(duration_s, 2)),
        )

    def summary(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT interval, status, COUNT(*) AS n, SUM(rows) AS bars FROM fetch_state GROUP BY interval, status "
            "ORDER BY interval, status")
        return [dict(r) for r in rows]

    def close(self) -> None:
        self.conn.close()
