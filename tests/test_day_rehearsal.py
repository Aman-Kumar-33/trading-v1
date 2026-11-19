"""Dress rehearsal: real recorded trading days (Angel 1-minute bars, replayed as ticks + closed candles) run through BOTH systems exactly
as run_trading.py wires them - paper broker, order manager, trade journal and trades database, strategy runner, force-exit commands,
5% stop, square-off - using the real config files. Every day must end flat by 15:00, no order may fall outside 09:15-15:00, every
trade must be in the database (read through a second connection, as the dashboard does), and a force-exit must close its position.

Run from second-system/:  python -m pytest tests -q
"""
from __future__ import annotations

import asyncio
import sqlite3
import sys
from datetime import datetime, time, timedelta
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.brokers.paper import PaperBroker  # noqa: E402
from backend.core import events  # noqa: E402
from backend.core.clock import IST  # noqa: E402
from backend.core.config import load_settings  # noqa: E402
from backend.core.events import EventBus  # noqa: E402
from backend.core.models import Candle, Mode, Tick  # noqa: E402
from backend.data import history_store  # noqa: E402
from backend.execution import process_control as pc  # noqa: E402
from backend.execution.journal import TradeJournal  # noqa: E402
from backend.execution.order_manager import OrderManager  # noqa: E402
from backend.execution.strategy_runner import SimpleRunner  # noqa: E402
from backend.notify import telegram as tg  # noqa: E402
from backend.research import live_macd_replay as R  # noqa: E402
from backend.risk.risk_manager import RiskLimits, RiskManager  # noqa: E402
from backend.storage.db import init_db  # noqa: E402
from backend.strategies.intraday.macd_only import MacdOnly  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "research_data" / "angel_1m"
SYMBOLS = ["SBIN", "TCS", "RELIANCE"]
DAYS_PER_STOCK = 3
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="the recorded 1-minute data is not on this machine")


@pytest.fixture(autouse=True)
def _no_telegram(monkeypatch):
    monkeypatch.setattr(tg, "notify", lambda *_a, **_k: None)


def _hhmm(text):
    h, m = text.split(":")
    return time(int(h), int(m))


def _system(name):
    cfg_path = ROOT / "config" / ("settings.yaml" if name == "main" else "settings_5m.yaml")
    t = load_settings(cfg_path).section("trading")
    return {"name": name, "tf": t["intraday_timeframe"], "mode": t.get("signal_mode", "bar_close"),
            "tol": float(t.get("macd_cross_tolerance", 0.0)), "tol_atr": float(t.get("macd_cross_tolerance_atr", 0.0)),
            "stop": float(t.get("stop_loss_pct") or 0) or None, "trade_start": _hhmm(t["trade_start"]),
            "square_off": _hhmm(t["square_off"])}


def _sessions(symbol):
    """The stock's full recorded sessions, newest first, and its 1-minute frame."""
    df1 = R.load_1m(DATA / symbol)
    if df1.empty:
        return None, []
    rows = df1[~df1["synthetic"]].groupby(df1["ts"].dt.date).size()
    days = [d for d in sorted(rows.index) if rows[d] >= 358]
    return df1, days[-DAYS_PER_STOCK:]


def _bars_frame(df1, minutes):
    b = R.to_bars(df1, minutes)
    ts = [pd.Timestamp(d).tz_localize(IST) + pd.Timedelta(minutes=int(m)) for d, m in zip(b["day"], b["start_min"])]
    return pd.DataFrame({"ts": ts, "open": b["open"].astype(float), "high": b["high"].astype(float), "low": b["low"].astype(float),
                         "close": b["close"].astype(float), "volume": 1000.0})


def _ticks(row, ts0):
    """Four ticks inside one recorded minute: open, the extreme on the way, the other extreme, close."""
    o, h, l, c = row["open"], row["high"], row["low"], row["close"]
    path = [o, h, l, c] if c < o else [o, l, h, c]
    return [Tick(row["symbol"], ts0 + timedelta(seconds=s), float(px)) for s, px in zip((0, 20, 40, 59), path)]


async def _run_day(system, symbol, df1, day, force_exit_after: int | None = None, tmp: Path | None = None):
    """One session through one system. Returns (order manager, db path, signal log, force-exit result)."""
    bus = EventBus()
    paper = PaperBroker(bus, starting_cash=1_000_000)
    await paper.connect()
    db = tmp / f"{system['name']}_{symbol}_{day}.db"
    conn = init_db(db)
    journal = TradeJournal(conn, tmp / f"open_{system['name']}_{symbol}_{day}.json", Mode.PAPER)
    om = OrderManager(paper, RiskManager(RiskLimits(capital=1_000_000)), mode=Mode.PAPER, square_off_time=system["square_off"],
                      carry_longs=False, journal=journal, profit_lock_stop=False, trail_stop=False, stop_loss_pct=system["stop"])
    log = []
    runner = SimpleRunner(bus, {system["tf"]: [MacdOnly(version="v1")]}, {}, om, capital=1_000_000, history_bars=500,
                          signal_sink=lambda sig, status, why: log.append((sig.side.value, str(sig.ts), status, why)),
                          watchlist={symbol}, square_off_time=system["square_off"], signal_mode=system["mode"],
                          tolerance=system["tol"], tolerance_atr=system["tol_atr"], trade_start=system["trade_start"],
                          stop_loss_pct=system["stop"])
    minutes = int(system["tf"][:-1])
    before = df1[df1["ts"].dt.date < day]
    history = history_store.settle_closing_bars(_bars_frame(before, minutes), system["tf"])
    runner.set_history(symbol, system["tf"], history)

    today = df1[(df1["ts"].dt.date == day) & (~df1["synthetic"])].copy()
    today["symbol"] = symbol
    bars = R.to_bars(today, minutes)
    bar_end = {(int(r.start_min)): int(r.end_min) for r in bars.itertuples()}
    bar_of = {int(r.start_min): r for r in bars.itertuples()}
    midnight = pd.Timestamp(day).tz_localize(IST).to_pydatetime()
    forced = None
    for _, row in today.iterrows():
        ts0 = row["ts"].to_pydatetime()
        for tick in _ticks(row, ts0):
            await bus.publish(events.TICK, tick)
        mod = ts0.hour * 60 + ts0.minute
        # a candle closes right after the last minute inside it
        for start, end in bar_end.items():
            if mod + 1 == end:
                r = bar_of[start]
                c = Candle(symbol, system["tf"], midnight + timedelta(minutes=start), midnight + timedelta(minutes=end),
                           float(r.open), float(r.high), float(r.low), float(r.close), 1000, 4, True)
                await bus.publish(events.CANDLE_CLOSED, c)
        # the force-exit button: queued by the dashboard, picked up by the trading loop's once-a-second poll
        if force_exit_after is not None and forced is None and om.open_positions:
            held = om.open_positions[0]
            if (ts0 - held.opened_at).total_seconds() >= force_exit_after * 60 and ts0.time() < system["square_off"]:
                pc.queue_close_position(_Settings(tmp), "paper", held.symbol)
                for cmd in pc.pop_commands(_Settings(tmp), "paper"):
                    if cmd.get("action") == "close_position":
                        await om.close_manual(cmd["symbol"], ts0)
                forced = {"symbol": held.symbol, "at": ts0, "still_open": om.has_position(held.symbol)}
    conn.close()
    return om, db, log, forced


class _Settings:
    """Just enough of Settings for the command queue: a state folder."""
    def __init__(self, folder: Path):
        self.paths = type("P", (), {"live_data": folder})()


def _check_day(system, symbol, day, om, db):
    start = datetime.combine(day, system["trade_start"], IST)
    off = datetime.combine(day, system["square_off"], IST)
    assert om.open_positions == [], f"{system['name']} {symbol} {day}: positions left open at the end of the day"
    for t in om.closed_trades:
        opened = t.position.opened_at
        assert start <= opened < off, f"order at {opened:%H:%M:%S} is outside {start:%H:%M}-{off:%H:%M}"
        assert t.closed_at <= off + timedelta(minutes=1) and t.closed_at > opened
        assert t.exit_reason in {"SIGNAL", "STOP", "SQUARE_OFF", "MANUAL"}, t.exit_reason
        assert t.position.qty >= 1 and t.position.entry_price > 0 and t.costs > 0
        assert t.position.side.value == "BUY"                                             # long/flat: never a short
        if t.exit_reason == "SQUARE_OFF":
            assert t.closed_at >= off - timedelta(seconds=1)
    # what the dashboard sees: a second connection, after everything was written
    other = sqlite3.connect(db)
    rows = other.execute("select symbol, side, entry_ts, exit_ts, exit_reason, pnl_net from trades").fetchall()
    other.close()
    assert len(rows) == len(om.closed_trades), "a closed trade is missing from the trades table"
    assert all(r[3] and r[4] and r[5] is not None for r in rows)


@pytest.mark.parametrize("name", ["main", "5m"])
def test_real_days_run_clean_through_the_whole_system(name, tmp_path):
    system = _system(name)
    assert system["tf"] == ("15m" if name == "main" else "5m")
    total_trades, exits = 0, set()
    for symbol in SYMBOLS:
        df1, days = _sessions(symbol)
        if df1 is None:
            continue
        for day in days:
            om, db, _log, _ = asyncio.run(_run_day(system, symbol, df1, day, tmp=tmp_path))
            _check_day(system, symbol, day, om, db)
            total_trades += len(om.closed_trades)
            exits |= {t.exit_reason for t in om.closed_trades}
    assert total_trades > 0, "the rehearsal never traded - nothing was proven"
    assert "SIGNAL" in exits                                                                # normal cross exits happen...
    # ...and the day always ends flat: every position that was still open at 15:00 was squared off, not carried
    assert exits <= {"SIGNAL", "STOP", "SQUARE_OFF"}


@pytest.mark.parametrize("name", ["main", "5m"])
def test_force_exit_closes_the_position_and_the_day_still_ends_clean(name, tmp_path):
    system = _system(name)
    done = 0
    for symbol in SYMBOLS:
        df1, days = _sessions(symbol)
        if df1 is None:
            continue
        for day in days:
            om, db, _log, forced = asyncio.run(_run_day(system, symbol, df1, day, force_exit_after=5, tmp=tmp_path))
            _check_day(system, symbol, day, om, db)
            if forced:
                assert forced["still_open"] is False
                manual = [t for t in om.closed_trades if t.exit_reason == "MANUAL"]
                assert len(manual) == 1 and manual[0].position.symbol == forced["symbol"]
                assert manual[0].closed_at == forced["at"]
                done += 1
    assert done > 0, "no day had an open position to force-exit"
