"""Intraday only: every position is sold at the 15:00 square-off, no new buys are taken at/after it, and a buy from
the 15:15 closing bar is not carried to the next morning.

Run from second-system/:  python -m pytest tests -q
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_macd_system import (PER_DAY, build_system, make_closes, run_candles, session_starts)  # noqa: E402

from backend.core.clock import IST  # noqa: E402
from backend.core.models import Side, Signal, Horizon  # noqa: E402
from backend.execution.order_manager import Position  # noqa: E402
from backend.notify import telegram as tg  # noqa: E402
from backend.strategies.intraday.macd_only import MacdOnly  # noqa: E402

SQUARE_OFF = time(15, 0)


@pytest.fixture(autouse=True)
def _no_telegram(monkeypatch):
    monkeypatch.setattr(tg, "notify", lambda *_: None)


def _intraday_system(n=300, warm=100):
    closes = make_closes(n)
    starts = session_starts(n // PER_DAY + 1)[:n]
    bus, paper, om, runner, log = build_system(closes, starts, warm)
    om.carry_longs = False
    om.square_off_time = SQUARE_OFF
    runner.square_off_time = SQUARE_OFF
    return bus, paper, om, runner, log, closes, starts


def test_no_buy_is_taken_at_or_after_the_square_off_time():
    bus, paper, om, runner, log, closes, starts = _intraday_system()
    run_candles(bus, paper, closes, starts, 100, 300)
    bought = [t.position.opened_at for t in om.closed_trades] + [p.opened_at for p in om.open_positions]
    assert bought, "the seed has bullish crosses, so some buys must happen"
    assert all(t.time() < SQUARE_OFF for t in bought)
    assert all(p.product == "MIS" for p in om.open_positions)             # intraday-only longs are MIS, not CNC
    # a buy signal on the bar that closes at 15:00 (or later) is reported as skipped, never bought
    sig = Signal(symbol="NEW", side=Side.BUY, horizon=Horizon.INTRADAY, stop_loss=1.0, strategy_id=MacdOnly.strategy_id,
                 strategy_version="v1", ts=datetime(2026, 11, 17, 14, 45, tzinfo=IST), timeframe="15m", entry=1000.0, reason="x")
    before = len(om.open_positions) + len(om.closed_trades)
    asyncio.run(runner._act("NEW", sig, datetime(2026, 11, 17, 15, 0, tzinfo=IST), 1000.0))
    assert len(om.open_positions) + len(om.closed_trades) == before
    assert any("no new buys at/after the 15:00 square-off" in (why or "") for _, _, _, why in log)


def test_every_open_position_is_sold_at_the_square_off_time():
    bus, paper, om, runner, log, closes, starts = _intraday_system()
    day = datetime(2026, 11, 17, tzinfo=IST)
    for sym, px in (("AAA", 100.0), ("BBB", 250.0)):
        paper.set_price(sym, px)
        om.open_positions.append(Position(sym, Side.BUY, Horizon.INTRADAY, 10, px, px - 50, None, "macd_only", "v1", None,
                                          day.replace(hour=10), product="MIS"))

    async def go(h, m):
        return await om.on_time(day.replace(hour=h, minute=m))

    assert asyncio.run(go(14, 59)) == [] and len(om.open_positions) == 2   # one minute before: nothing happens
    closed = asyncio.run(go(15, 0))
    assert sorted(t.position.symbol for t in closed) == ["AAA", "BBB"]
    assert {t.exit_reason for t in closed} == {"SQUARE_OFF"} and not om.open_positions


def test_a_buy_from_the_closing_bar_is_not_carried_to_next_morning(tmp_path):
    bus, paper, om, runner, log, closes, starts = _intraday_system()
    sig = Signal(symbol="TEST", side=Side.BUY, horizon=Horizon.INTRADAY, stop_loss=1.0, strategy_id=MacdOnly.strategy_id,
                 strategy_version="v1", ts=datetime(2026, 11, 16, 15, 15, tzinfo=IST), timeframe="15m", entry=1000.0, reason="x")
    runner._deferred["TEST"] = (sig, sig.ts.date())
    from backend.core.models import Tick
    paper.set_price("TEST", 1000.0)
    asyncio.run(runner.on_tick(Tick("TEST", datetime(2026, 11, 17, 9, 15, 5, tzinfo=IST), 1000.0)))
    assert not om.open_positions and "TEST" not in runner._deferred
    assert any("previous day's 15:15 bar" in (why or "") for _, _, _, why in log)
