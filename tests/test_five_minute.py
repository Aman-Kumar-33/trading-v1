"""The "5 minute" system: MACD computed from live ticks with a tolerance, a settings file that extends the main one, and
a stock list that follows the main system's.

Run from second-system/:  python -m pytest tests -q
"""
from __future__ import annotations

import asyncio
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_macd_system import (PER_DAY, TF, build_system, make_closes, reference_cross_bars, session_starts)  # noqa: E402

from backend.core import events  # noqa: E402
from backend.core.clock import IST  # noqa: E402
from backend.core.config import load_settings  # noqa: E402
from backend.core.models import Tick  # noqa: E402
from backend.data import trending, watchlist  # noqa: E402
from backend.execution.strategy_runner import _LiveMacd, _bucket_start  # noqa: E402
from backend.notify import telegram as tg  # noqa: E402


@pytest.fixture(autouse=True)
def _no_telegram(monkeypatch):
    monkeypatch.setattr(tg, "notify", lambda *_: None)


# ---------------------------------------------------------------------- live (tick-driven) signals
def _live_system(tolerance: float, closes=None, warm=100, n=300):
    closes = make_closes(n) if closes is None else closes
    starts = session_starts(n // PER_DAY + 1)[:n]
    bus, paper, om, runner, log = build_system(closes, starts, warm)
    runner.signal_mode, runner.tolerance = "live", tolerance
    return bus, paper, om, runner, log, closes, starts


def _run_ticks(bus, paper, runner, closes, starts, begin, end, ticks_per_bar=1):
    """Each bar: ticks inside it (the last one at the bar's close price), then the bar closes."""
    from test_macd_system import candle

    async def go():
        for i in range(begin, end):
            for k in range(ticks_per_bar):
                ts = starts[i] + timedelta(minutes=10 + k)
                px = float(closes[i])
                paper.set_price("TEST", px)
                await bus.publish(events.TICK, Tick("TEST", ts, px))
            await bus.publish(events.CANDLE_CLOSED, candle("TEST", starts[i], float(closes[i])))
    asyncio.run(go())


def test_live_mode_with_zero_tolerance_buys_on_the_same_bars_as_the_candle_close_rule():
    bus, paper, om, runner, log, closes, starts = _live_system(0.0)
    _run_ticks(bus, paper, runner, closes, starts, 100, 300)
    ref_bull, _ = reference_cross_bars(closes)
    bought = sorted(t.position.opened_at for t in om.closed_trades) + [p.opened_at for p in om.open_positions]
    got = [i for i in range(100, 300) if any(starts[i] <= t < starts[i] + timedelta(minutes=15) for t in bought)]
    skip = 130
    assert [i for i in got if i >= skip] == [i for i in ref_bull if i >= skip]
    assert all(p.side.value == "BUY" for p in om.open_positions)


def test_a_bigger_tolerance_gives_fewer_trades():
    counts = []
    for tol in (0.0, 0.5, 5.0):
        bus, paper, om, runner, log, closes, starts = _live_system(tol)
        _run_ticks(bus, paper, runner, closes, starts, 100, 300)
        counts.append(len(om.closed_trades) + len(om.open_positions))
    assert counts[0] >= counts[1] >= counts[2] and counts[0] > counts[2]


def _seed_base(runner, tick_ts, state="down"):
    """EMAs that put MACD exactly on the signal line, as of the closed bar before the one `tick_ts` falls in."""
    bucket = _bucket_start(tick_ts, TF)
    runner._live["TEST"] = _LiveMacd(last_start=bucket - timedelta(minutes=15), e12=100.0, e26=100.0, sig=0.0, gap=0.0,
                                     atr=1.0, state=state)


def test_wobbles_inside_the_tolerance_band_do_nothing_and_crossing_it_trades():
    bus, paper, om, runner, log, closes, starts = _live_system(0.02)
    t0 = datetime(2026, 10, 6, 10, 20, 30, tzinfo=IST)
    _seed_base(runner, t0)

    def tick(sec: int, price: float):
        paper.set_price("TEST", price)
        asyncio.run(bus.publish(events.TICK, Tick("TEST", t0 + timedelta(seconds=sec), price)))

    # price 100.2 -> gap +0.0128, inside the +-0.02 band
    tick(0, 100.2)
    assert not om.open_positions
    tick(1, 99.8)                                    # and -0.0128 on the other side: still nothing
    assert not om.open_positions and not log
    tick(2, 100.4)                                   # gap +0.0255 -> above the tolerance: BUY at once
    assert om.has_position("TEST") and om.open_positions[0].opened_at == t0 + timedelta(seconds=2)
    tick(3, 100.2)                                   # back inside the band: the long is kept
    assert om.has_position("TEST")
    tick(4, 99.6)                                    # gap -0.0255 -> below -tolerance: SELL
    assert not om.has_position("TEST") and om.closed_trades[-1].exit_reason == "SIGNAL"
    assert [row[0] for row in log] == ["BUY", "SELL"]
    assert [row[2] for row in log] == ["ACCEPTED", "ACCEPTED"]


def test_no_buy_when_already_above_the_line_without_a_fresh_cross():
    bus, paper, om, runner, log, closes, starts = _live_system(0.02)
    t0 = datetime(2026, 10, 6, 10, 20, 30, tzinfo=IST)
    _seed_base(runner, t0, state="up")               # already above the line when the system started watching
    paper.set_price("TEST", 100.5)
    asyncio.run(bus.publish(events.TICK, Tick("TEST", t0, 100.5)))
    assert not om.open_positions and not log


def test_bucket_start_is_aligned_to_the_open():
    ts = datetime(2026, 10, 6, 11, 7, 41, tzinfo=IST)
    assert _bucket_start(ts, "5m") == datetime(2026, 10, 6, 11, 5, tzinfo=IST)
    assert _bucket_start(ts, "15m") == datetime(2026, 10, 6, 11, 0, tzinfo=IST)
    assert _bucket_start(datetime(2026, 10, 6, 9, 15, 0, tzinfo=IST), "5m") == datetime(2026, 10, 6, 9, 15, tzinfo=IST)


# ---------------------------------------------------------------------- settings that extend another file; a list that follows
def _pair(tmp_path):
    main = tmp_path / "main.yaml"
    main.write_text(f"paths:\n  live_data: {(tmp_path / 'data').as_posix()}\n  logs: {(tmp_path / 'logs').as_posix()}\n"
                    "trading:\n  intraday_timeframe: 15m\n  paper_capital: 1000000\n  symbols: []\n", encoding="utf-8")
    five = tmp_path / "five.yaml"
    five.write_text("extends: main.yaml\n"
                    f"instance:\n  name: 5 minute\n  follow: {main.as_posix()}\n"
                    f"paths:\n  live_data: {(tmp_path / 'data5').as_posix()}\n"
                    "trading:\n  intraday_timeframe: 5m\n  macd_cross_tolerance: 0.02\n", encoding="utf-8")
    return load_settings(main), load_settings(five)


def test_settings_extend_the_main_file_and_keep_separate_folders(tmp_path):
    main, five = _pair(tmp_path)
    assert five.section("trading")["intraday_timeframe"] == "5m" and five.section("trading")["paper_capital"] == 1_000_000
    assert main.section("trading")["intraday_timeframe"] == "15m"
    assert five.paths.live_data != main.paths.live_data and five.paths.db_file != main.paths.db_file
    assert five.paths.logs == main.paths.logs          # (logs were not overridden in this test file)


def test_the_5_minute_system_trades_the_main_systems_stocks(tmp_path):
    main, five = _pair(tmp_path)
    assert not watchlist.is_auto_mode(five, "paper") and watchlist.active_symbols(five, "paper")[0] == []   # nothing picked yet
    trending.save_picks(main, "paper", [{"rank": 1, "symbol": "SBIN", "score": 80.0}, {"rank": 2, "symbol": "TCS", "score": 70.0}],
                        datetime.now().date(), "15m")
    syms, source = watchlist.active_symbols(five, "paper")
    assert syms == ["SBIN", "TCS"] and "main system" in source
    before = watchlist.change_stamp(five, "paper")
    watchlist.save(main, "paper", ["itc"])            # the main system's hand-given list wins there, so it wins here
    assert watchlist.active_symbols(five, "paper")[0] == ["ITC"] and watchlist.change_stamp(five, "paper") != before
    watchlist.save(five, "paper", ["wipro"])          # its own box is ignored: it never trades anything else
    assert watchlist.active_symbols(five, "paper")[0] == ["ITC"]


# ---------------------------------------------------------------------- tolerance scaled to the stock's ATR, and a hold time
def test_the_cross_must_hold_before_the_order_is_placed():
    bus, paper, om, runner, log, closes, starts = _live_system(0.02)
    runner.hold_seconds = 60
    t0 = datetime(2026, 10, 6, 10, 20, 0, tzinfo=IST)
    _seed_base(runner, t0)

    def tick(sec: int, price: float):
        paper.set_price("TEST", price)
        asyncio.run(bus.publish(events.TICK, Tick("TEST", t0 + timedelta(seconds=sec), price)))

    tick(0, 100.4)                       # above the tolerance, but only just started
    tick(30, 100.4)
    assert not om.open_positions
    tick(40, 100.2)                      # fell back into the band: the cross did not hold, the clock restarts
    tick(50, 100.4)
    tick(100, 100.4)                     # only 50 s since it re-crossed
    assert not om.open_positions
    tick(111, 100.4)                     # 61 s -> now it is real
    assert om.has_position("TEST") and om.open_positions[0].opened_at == t0 + timedelta(seconds=111)
    tick(120, 99.6)                      # the exit also has to hold
    assert om.has_position("TEST")
    tick(181, 99.6)
    assert not om.has_position("TEST")


def test_tolerance_scales_with_the_stocks_atr():
    bus, paper, om, runner, log, closes, starts = _live_system(0.02)
    runner.tolerance_atr = 0.1           # tolerance = max(0.02, 0.1 x ATR); the seeded ATR is 1.0 -> 0.1
    t0 = datetime(2026, 10, 6, 10, 20, 0, tzinfo=IST)
    _seed_base(runner, t0)
    paper.set_price("TEST", 100.4)       # gap +0.0255: enough for a fixed 0.02, not for 0.1
    asyncio.run(bus.publish(events.TICK, Tick("TEST", t0, 100.4)))
    assert not om.open_positions
    paper.set_price("TEST", 102.0)       # gap +0.128 > 0.1
    asyncio.run(bus.publish(events.TICK, Tick("TEST", t0 + timedelta(seconds=1), 102.0)))
    assert om.has_position("TEST")
