"""Trading starts at 09:15: the day's trending list is picked before the open (history loading 09:00-09:20), and a live
signal can fire on the very first ticks of the session.

Run from second-system/:  python -m pytest tests -q
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_five_minute import _live_system  # noqa: E402,F401

from backend.core import events  # noqa: E402
from backend.core.clock import IST  # noqa: E402
from backend.core.models import Tick  # noqa: E402
from backend.data import trending  # noqa: E402
from backend.execution.strategy_runner import _LiveMacd  # noqa: E402


def at(h: int, m: int, s: int = 0) -> datetime:
    return datetime(2026, 10, 7, h, m, s, tzinfo=IST)   # a Wednesday


# ---------------------------------------------------------------------- when the list is picked
def test_picks_as_soon_as_the_whole_index_history_is_loaded_even_before_the_open():
    assert trending.pick_plan(at(9, 6), at(9, 0), loaded=50, total=50) == (True, False)


def test_waits_for_missing_history_until_09_20_then_ranks_what_has_loaded():
    started = at(9, 0)
    assert trending.pick_plan(at(9, 10), started, loaded=44, total=50) == (False, False)
    assert trending.pick_plan(at(9, 19, 59), started, loaded=44, total=50) == (False, False)
    assert trending.pick_plan(at(9, 20), started, loaded=44, total=50) == (True, False)


def test_never_picks_before_09_00():
    assert trending.pick_plan(at(8, 30), at(8, 29), loaded=50, total=50) == (False, False)


def test_before_09_30_the_pick_uses_history_only_after_that_today_s_quote_too():
    assert trending.pick_plan(at(9, 14), at(9, 0), 50, 50)[1] is False
    assert trending.pick_plan(at(9, 30), at(9, 0), 50, 50)[1] is True


def test_a_late_start_still_gets_time_to_load_the_index():
    started = at(10, 0)
    assert trending.pick_plan(at(10, 3), started, loaded=20, total=50) == (False, False)
    assert trending.pick_plan(at(10, 8), started, loaded=20, total=50) == (True, True)
    assert trending.pick_plan(at(10, 3), started, loaded=50, total=50) == (True, True)


def test_no_picks_late_in_the_day():
    assert trending.pick_plan(at(15, 20), at(9, 0), 50, 50) == (False, False)


# ---------------------------------------------------------------------- signals from the first tick of the session
def test_live_signal_fires_on_the_first_ticks_of_the_session():
    bus, paper, om, runner, log, closes, starts = _live_system(0.02)
    # base = yesterday's last bar (15:15 on a 15m system); the forming bar is today's 09:15 one
    runner._live["TEST"] = _LiveMacd(last_start=at(15, 15) - timedelta(days=1), e12=100.0, e26=100.0, sig=0.0, gap=0.0,
                                     atr=1.0, state="down")
    paper.set_price("TEST", 100.4)
    asyncio.run(bus.publish(events.TICK, Tick("TEST", at(9, 15, 3), 100.4)))
    assert om.has_position("TEST") and om.open_positions[0].opened_at == at(9, 15, 3)


def test_no_signal_from_pre_open_auction_ticks():
    bus, paper, om, runner, log, closes, starts = _live_system(0.02)
    runner._live["TEST"] = _LiveMacd(last_start=at(15, 15) - timedelta(days=1), e12=100.0, e26=100.0, sig=0.0, gap=0.0,
                                     atr=1.0, state="down")
    paper.set_price("TEST", 100.4)
    asyncio.run(bus.publish(events.TICK, Tick("TEST", at(9, 10), 100.4)))
    assert not om.open_positions
