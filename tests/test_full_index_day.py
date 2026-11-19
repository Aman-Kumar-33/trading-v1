"""Dress rehearsal of a whole day with EVERY NIFTY 50 stock at once, as both systems now run: one paper broker, one order manager (shared
cash, no limit on positions), one strategy runner watching all the stocks, real recorded Angel 1-minute bars replayed as ticks and
closed candles, force-exits, the 5% stop (5m system) and the 15:00 square-off. The day must end flat, every order must fall inside
09:15-15:00, the cash must never go below zero, and every trade must be in the trades table.

Run from second-system/:  python -m pytest tests -q
"""
from __future__ import annotations

import asyncio
import sqlite3
import sys
import time as wallclock
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_day_rehearsal import DATA, _Settings, _bars_frame, _system, _ticks  # noqa: E402

from backend.brokers.paper import PaperBroker  # noqa: E402
from backend.core import events  # noqa: E402
from backend.core.clock import IST  # noqa: E402
from backend.core.events import EventBus  # noqa: E402
from backend.core.models import Candle, Mode  # noqa: E402
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

pytestmark = pytest.mark.skipif(not DATA.exists(), reason="the recorded 1-minute data is not on this machine")
CAPITAL = 1_000_000


@pytest.fixture(autouse=True)
def _no_telegram(monkeypatch):
    monkeypatch.setattr(tg, "notify", lambda *_a, **_k: None)


def _load_index():
    """{symbol: 1-minute frame} for every recorded stock, and the newest day on which at least 80% of them have a full session."""
    frames = {}
    for d in sorted(DATA.iterdir()):
        if not d.is_dir():
            continue
        df1 = R.load_1m(d)
        if not df1.empty:
            frames[d.name.replace("_AND_", "&")] = df1
    full = {}
    for sym, df1 in frames.items():
        rows = df1[~df1["synthetic"]].groupby(df1["ts"].dt.date).size()
        for day in rows.index[rows >= 358]:
            full.setdefault(day, []).append(sym)
    days = [d for d in sorted(full) if len(full[d]) >= 0.8 * len(frames)]
    return frames, (days[-1] if days else None), full


async def _run_index_day(system, frames, day, symbols, tmp: Path, force_exit_every: int | None = None):
    bus = EventBus()
    paper = PaperBroker(bus, starting_cash=CAPITAL)
    await paper.connect()
    db = tmp / f"index_{system['name']}.db"
    conn = init_db(db)
    journal = TradeJournal(conn, tmp / f"open_{system['name']}.json", Mode.PAPER)
    om = OrderManager(paper, RiskManager(RiskLimits(capital=CAPITAL)), mode=Mode.PAPER, square_off_time=system["square_off"],
                      carry_longs=False, journal=journal, profit_lock_stop=False, trail_stop=False, stop_loss_pct=system["stop"])
    log = []
    runner = SimpleRunner(bus, {system["tf"]: [MacdOnly(version="v1")]}, {}, om, capital=CAPITAL, history_bars=500,
                          signal_sink=lambda sig, status, why: log.append((sig.symbol, sig.side.value, str(sig.ts), status, why)),
                          watchlist=set(symbols), square_off_time=system["square_off"], signal_mode=system["mode"],
                          tolerance=system["tol"], tolerance_atr=system["tol_atr"], trade_start=system["trade_start"],
                          stop_loss_pct=system["stop"], max_open_positions=0)
    minutes = int(system["tf"][:-1])
    midnight = pd.Timestamp(day).tz_localize(IST).to_pydatetime()
    per_symbol = {}
    for sym in symbols:
        df1 = frames[sym]
        before = df1[(df1["ts"].dt.date < day) & (df1["ts"].dt.date >= day - timedelta(days=40))]
        runner.set_history(sym, system["tf"], history_store.settle_closing_bars(_bars_frame(before, minutes), system["tf"]))
        today = df1[(df1["ts"].dt.date == day) & (~df1["synthetic"])].copy()
        today["symbol"] = sym
        bars = R.to_bars(today, minutes)
        per_symbol[sym] = ({int(r.start_min): int(r.end_min) for r in bars.itertuples()}, {int(r.start_min): r for r in bars.itertuples()})
    # one timeline for all stocks: minute by minute, every stock's rows of that minute
    timeline = {}
    for sym in symbols:
        df1 = frames[sym]
        today = df1[(df1["ts"].dt.date == day) & (~df1["synthetic"])]
        for _, row in today.assign(symbol=sym).iterrows():
            timeline.setdefault(row["ts"].to_pydatetime(), []).append(row)
    forced, peak_open, worst_cash, closes = [], 0, CAPITAL, 0
    started = wallclock.perf_counter()
    for ts0 in sorted(timeline):
        mod = ts0.hour * 60 + ts0.minute
        for row in timeline[ts0]:
            for tick in _ticks(row, ts0):
                await bus.publish(events.TICK, tick)
        for row in timeline[ts0]:
            bar_end, bar_of = per_symbol[row["symbol"]]
            for start, end in bar_end.items():
                if mod + 1 == end:
                    r = bar_of[start]
                    await bus.publish(events.CANDLE_CLOSED, Candle(row["symbol"], system["tf"], midnight + timedelta(minutes=start),
                                                                     midnight + timedelta(minutes=end), float(r.open), float(r.high),
                                                                     float(r.low), float(r.close), 1000, 4, True))
                    closes += 1
        peak_open = max(peak_open, len(om.open_positions))
        worst_cash = min(worst_cash, CAPITAL + om.realized_pnl() - om.invested())
        # the force-exit button, every `force_exit_every` minutes on whatever position has been open the longest
        ripe = [x for x in om.open_positions if (ts0 - x.opened_at).total_seconds() >= 300]       # held for at least 5 minutes
        if force_exit_every and mod % force_exit_every == 0 and ripe and ts0.time() < system["square_off"]:
            held = ripe[0]
            pc.queue_close_position(_Settings(tmp), "paper", held.symbol)
            for cmd in pc.pop_commands(_Settings(tmp), "paper"):
                await om.close_manual(cmd["symbol"], ts0)
            forced.append(held.symbol)
    conn.close()
    return om, db, log, {"peak_open": peak_open, "worst_cash": worst_cash, "forced": forced, "closes": closes,
                         "seconds": wallclock.perf_counter() - started}


def _check(system, day, om, db):
    start = datetime.combine(day, system["trade_start"], IST)
    off = datetime.combine(day, system["square_off"], IST)
    assert om.open_positions == [], f"{system['name']}: positions left open at the end of the day"
    for t in om.closed_trades:
        assert start <= t.position.opened_at < off, f"order at {t.position.opened_at:%H:%M:%S} outside {start:%H:%M}-{off:%H:%M}"
        assert t.closed_at > t.position.opened_at and t.closed_at <= off + timedelta(minutes=1)
        assert t.exit_reason in {"SIGNAL", "STOP", "SQUARE_OFF", "MANUAL"}, t.exit_reason
        assert t.position.side.value == "BUY" and t.position.qty >= 1 and t.costs > 0
    other = sqlite3.connect(db)
    rows = other.execute("select symbol, entry_ts, exit_ts, exit_reason, pnl_net from trades").fetchall()
    other.close()
    assert len(rows) == len(om.closed_trades) and all(r[2] and r[3] and r[4] is not None for r in rows)


@pytest.mark.parametrize("name", ["main", "5m"])
def test_a_whole_index_day_runs_clean(name, tmp_path):
    frames, day, full = _load_index()
    if day is None:
        pytest.skip("no recorded day with most of the index")
    symbols = sorted(full[day])
    assert len(symbols) >= 40, "the recording should cover (nearly) the whole index"
    system = _system(name)
    om, db, log, info = asyncio.run(_run_index_day(system, frames, day, symbols, tmp_path))
    _check(system, day, om, db)
    assert len(om.closed_trades) > 0
    assert info["worst_cash"] >= -1.0, f"cash went negative: {info['worst_cash']:,.0f}"       # shared cash is never overspent
    traded = {t.position.symbol for t in om.closed_trades}
    assert len(traded) > 3, "with the whole index watched, many different stocks must trade"
    # without a position limit, nothing was ever rejected for "max open positions"
    assert not any("max open positions" in str(w) for *_x, w in log)
    print(f"\n{name}: {day} {len(symbols)} stocks, {len(om.closed_trades)} trades in {len(traded)} stocks, peak {info['peak_open']} open at once, "
          f"min free cash {info['worst_cash']:,.0f}, {info['closes']} candles in {info['seconds']:.1f}s, "
          f"exits {sorted({t.exit_reason for t in om.closed_trades})}, cash skips {sum('free cash' in str(w) for *_x, w in log)}")


@pytest.mark.parametrize("name", ["main", "5m"])
def test_a_whole_index_day_with_force_exits_ends_clean(name, tmp_path):
    frames, day, full = _load_index()
    if day is None:
        pytest.skip("no recorded day with most of the index")
    system = _system(name)
    om, db, log, info = asyncio.run(_run_index_day(system, frames, day, sorted(full[day])[:20], tmp_path, force_exit_every=7))
    _check(system, day, om, db)
    manual = [t for t in om.closed_trades if t.exit_reason == "MANUAL"]
    assert len(manual) == len(info["forced"]) > 0                                              # every force-exit closed its position
    print(f"\n{name}: {len(manual)} force-exits among {len(om.closed_trades)} trades")
