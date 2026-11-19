"""SQLite database - single file (data/algo.db), no server to install or manage.

Holds the trading records (instruments, live candles, signals, orders, trades, positions, strategy
versions, journal, news, daily summaries). Bulk historical price data lives in Parquet files under
backtest_data/, not here.

WAL mode lets the dashboard read while the engine writes.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS instruments (
    symbol          TEXT PRIMARY KEY,        -- NSE symbol, e.g. RELIANCE
    exchange        TEXT NOT NULL DEFAULT 'NSE',
    name            TEXT,
    isin            TEXT,
    industry        TEXT,
    series          TEXT,
    yf_symbol       TEXT,                    -- RELIANCE.NS
    angel_token     TEXT,
    zerodha_token   TEXT,
    lot_size        INTEGER DEFAULT 1,
    tick_size       REAL DEFAULT 0.05,
    price_band      TEXT,                    -- circuit band: 2, 5, 10, 20 or 'No Band'
    in_fo_ban       INTEGER DEFAULT 0,
    active          INTEGER DEFAULT 1,
    updated_at      TEXT
);

-- Live / replay candles for recent sessions (history is in Parquet).
CREATE TABLE IF NOT EXISTS candles (
    symbol      TEXT NOT NULL,
    timeframe   TEXT NOT NULL,
    ts_start    TEXT NOT NULL,              -- ISO timestamp, IST
    ts_end      TEXT NOT NULL,
    open        REAL NOT NULL,
    high        REAL NOT NULL,
    low         REAL NOT NULL,
    close       REAL NOT NULL,
    volume      INTEGER NOT NULL DEFAULT 0,
    tick_count  INTEGER NOT NULL DEFAULT 0,
    source      TEXT NOT NULL DEFAULT 'live',   -- live | replay
    PRIMARY KEY (symbol, timeframe, ts_start, source)
);
CREATE INDEX IF NOT EXISTS idx_candles_tf_ts ON candles (timeframe, ts_start);

CREATE TABLE IF NOT EXISTS signals (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                TEXT NOT NULL,
    mode              TEXT NOT NULL,             -- BACKTEST | PAPER | LIVE
    strategy_id       TEXT NOT NULL,
    strategy_version  TEXT NOT NULL,
    symbol            TEXT NOT NULL,
    side              TEXT NOT NULL,             -- BUY | SELL
    horizon           TEXT NOT NULL,             -- INTRADAY | LONG_TERM
    timeframe         TEXT,
    entry             REAL,
    stop_loss         REAL NOT NULL,             -- every signal must carry a stop loss
    targets_json      TEXT,
    confidence        REAL,
    reason            TEXT,
    patterns_json     TEXT,
    regime            TEXT,
    status            TEXT NOT NULL DEFAULT 'NEW',   -- NEW | ACCEPTED | REJECTED | EXPIRED
    reject_reason     TEXT
);
CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals (ts);

CREATE TABLE IF NOT EXISTS orders (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    broker_order_id   TEXT,
    signal_id         INTEGER REFERENCES signals(id),
    ts                TEXT NOT NULL,
    mode              TEXT NOT NULL,
    symbol            TEXT NOT NULL,
    side              TEXT NOT NULL,
    qty               INTEGER NOT NULL,
    order_type        TEXT NOT NULL,
    product           TEXT NOT NULL,
    price             REAL,
    trigger_price     REAL,
    status            TEXT NOT NULL,
    filled_qty        INTEGER DEFAULT 0,
    avg_price         REAL,
    algo_tag          TEXT,
    updated_at        TEXT
);

CREATE TABLE IF NOT EXISTS trades (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id         INTEGER REFERENCES signals(id),
    mode              TEXT NOT NULL,
    strategy_id       TEXT NOT NULL,
    strategy_version  TEXT NOT NULL,
    symbol            TEXT NOT NULL,
    side              TEXT NOT NULL,
    horizon           TEXT NOT NULL,
    qty               INTEGER NOT NULL,
    entry_ts          TEXT NOT NULL,
    entry_price       REAL NOT NULL,
    exit_ts           TEXT,
    exit_price        REAL,
    stop_loss         REAL NOT NULL,
    target            REAL,
    pnl_gross         REAL,
    costs             REAL,
    pnl_net           REAL,
    r_multiple        REAL,
    mae               REAL,                     -- max adverse excursion
    mfe               REAL,                     -- max favourable excursion
    slippage          REAL,
    exit_reason       TEXT                      -- TARGET | STOP | TRAIL | SQUARE_OFF | SIGNAL | MANUAL
);
CREATE INDEX IF NOT EXISTS idx_trades_entry ON trades (entry_ts);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades (symbol);

CREATE TABLE IF NOT EXISTS positions (
    symbol      TEXT NOT NULL,
    mode        TEXT NOT NULL,
    horizon     TEXT NOT NULL,
    qty         INTEGER NOT NULL,
    avg_price   REAL NOT NULL,
    stop_loss   REAL,
    target      REAL,
    opened_at   TEXT,
    updated_at  TEXT,
    PRIMARY KEY (symbol, mode, horizon)
);

CREATE TABLE IF NOT EXISTS strategy_versions (
    strategy_id     TEXT NOT NULL,
    version         TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    status          TEXT NOT NULL,          -- RESEARCH | PAPER | LIVE | RETIRED | REJECTED
    params_json     TEXT,
    code_hash       TEXT,
    parent_version  TEXT,
    notes           TEXT,
    PRIMARY KEY (strategy_id, version)
);

CREATE TABLE IF NOT EXISTS journal (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    trade_id    INTEGER REFERENCES trades(id),
    event       TEXT NOT NULL,
    data_json   TEXT
);

CREATE TABLE IF NOT EXISTS news_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    source        TEXT,
    symbols_json  TEXT,
    headline      TEXT,
    url           TEXT,
    sentiment     TEXT,
    score         REAL,
    raw_json      TEXT
);

CREATE TABLE IF NOT EXISTS daily_summary (
    date          TEXT NOT NULL,
    mode          TEXT NOT NULL,
    trades        INTEGER DEFAULT 0,
    wins          INTEGER DEFAULT 0,
    losses        INTEGER DEFAULT 0,
    pnl_gross     REAL DEFAULT 0,
    costs         REAL DEFAULT 0,
    pnl_net       REAL DEFAULT 0,
    max_drawdown  REAL DEFAULT 0,
    frozen        INTEGER DEFAULT 0,       -- 1 once the day is closed and the EOD snapshot is written
    frozen_at     TEXT,
    data_json     TEXT,
    PRIMARY KEY (date, mode)
);

CREATE TABLE IF NOT EXISTS session_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    event       TEXT NOT NULL,
    data_json   TEXT
);
"""


def connect(path: Path | str) -> sqlite3.Connection:
    path = Path(path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30, isolation_level=None)  # autocommit; use transaction() for batches
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(path: Path | str) -> sqlite3.Connection:
    conn = connect(path)
    conn.executescript(SCHEMA)
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    conn.execute("BEGIN")
    try:
        yield conn
    except Exception:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")
