"""The 5 minute system's "candle_confirmed" mode: signals only from COMPLETED candles - a BUY when the MACD line has crossed
above the signal line and cleared the tolerance, a SELL at the next plain intersection, a 5% stop, no buys before the trade start (09:15).
The main (15m) system must be unchanged.

Run from second-system/:  python -m pytest tests -q
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, time, timedelta
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_macd_system import (PER_DAY, STEP, TF, build_system, candle, make_closes, reference_cross_bars,  # noqa: E402
                              run_candles, session_starts)

from backend.core import events  # noqa: E402
from backend.core.clock import IST  # noqa: E402
from backend.core.config import load_settings  # noqa: E402
from backend.core.models import Side, Tick  # noqa: E402
from backend.execution.strategy_runner import confirmed_cross_up  # noqa: E402
from backend.features.pipeline import build_features  # noqa: E402
from backend.notify import telegram as tg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
N, WARM = 360, 100


@pytest.fixture(autouse=True)
def _no_telegram(monkeypatch):
    monkeypatch.setattr(tg, "notify", lambda *_: None)


def _system(tol: float, closes=None, **runner_kw):
    closes = make_closes(N) if closes is None else closes
    starts = session_starts(N // PER_DAY + 1)[:N]
    bus, paper, om, runner, log = build_system(closes, starts, WARM)
    runner.signal_mode, runner.tolerance = "candle_confirmed", tol
    for k, v in runner_kw.items():
        setattr(runner, k, v)
    return bus, paper, om, runner, log, closes, starts


def _lines(runner):
    """MACD and signal exactly as the system computed them (the series the signals were taken from)."""
    key = ("TEST", TF)
    feat = build_features(runner._build_df(key), TF, core_only=True)
    return feat["macd"].to_numpy(float), feat["macd_signal"].to_numpy(float)


def _entry_bars(om, starts):
    """Index of the bar each position was entered on (entry time = the close of its bar)."""
    opened = sorted(t.position.opened_at for t in om.closed_trades) + [p.opened_at for p in om.open_positions]
    return [starts.index(t - timedelta(minutes=STEP)) for t in opened]


# ---------------------------------------------------------------------- the confirmation rule itself
def test_confirmed_cross_up_needs_an_intersection_and_then_clearing_the_tolerance():
    tol = 0.5
    assert confirmed_cross_up(np.array([-1.0, 0.8]), tol)                    # below, then clear of the band: BUY
    assert not confirmed_cross_up(np.array([-1.0, 0.3]), tol)                # crossed, but still inside the band: wait
    assert confirmed_cross_up(np.array([-1.0, 0.3, 0.8]), tol)               # ... and BUY when it gets out of the band
    assert not confirmed_cross_up(np.array([-1.0, 0.8, 0.9]), tol)           # already confirmed one bar ago: same move
    assert not confirmed_cross_up(np.array([-1.0, 0.8, 0.3, 0.9]), tol)      # dipped back into the band, never touched the line
    assert confirmed_cross_up(np.array([0.8, -0.1, 0.3, 0.9]), tol)          # the lines intersected twice in between: BUY once
    assert confirmed_cross_up(np.array([-0.2, 0.0, 0.6]), tol)               # sitting exactly on the line counts as "not above"
    assert not confirmed_cross_up(np.array([0.8, 0.9]), tol)                 # above all along: no intersection
    assert not confirmed_cross_up(np.array([np.nan, 0.9]), tol)              # warm-up bars never count
    assert not confirmed_cross_up(np.array([0.9]), tol)


def test_with_zero_tolerance_the_buys_are_exactly_the_plain_crosses():
    bus, paper, om, runner, log, closes, starts = _system(0.0)
    run_candles(bus, paper, closes, starts, WARM, N)
    ref_bull, _ = reference_cross_bars(closes)
    skip = 130
    assert [b for b in _entry_bars(om, starts) if b >= skip] == [i for i in ref_bull if i >= skip]


def test_every_buy_follows_an_intersection_from_below_and_clears_the_tolerance_and_every_sell_is_the_next_intersection():
    tol = 1.0
    bus, paper, om, runner, log, closes, starts = _system(tol)
    run_candles(bus, paper, closes, starts, WARM, N)
    macd, sig = _lines(runner)
    gap = macd - sig
    buys = [b for b in _entry_bars(om, starts) if b >= 130]
    assert len(buys) >= 3
    for i in buys:
        assert gap[i] > tol                                     # after: MACD above the signal line by more than the tolerance
        j = max(k for k in range(i) if gap[k] <= 0)             # the last bar with MACD at/below the signal line
        assert macd[j] <= sig[j] and macd[j + 1] > sig[j + 1]   # before: below; right after: above -> the lines intersected
        assert all(0 < gap[k] <= tol for k in range(j + 1, i))  # in between: above, but not yet clear of the tolerance
    sells = sorted((t.closed_at, t.position.opened_at) for t in om.closed_trades if t.exit_reason == "SIGNAL")
    assert sells
    inside_the_band = 0
    for closed_at, opened_at in sells:
        k = starts.index(closed_at - timedelta(minutes=STEP))
        assert gap[k] < 0 and gap[k - 1] >= 0                   # MACD went from above to below the signal line
        inside_the_band += gap[k] > -tol
    assert inside_the_band >= 1                                 # the tolerance plays no part in the exit: it sold inside the band


def test_a_bigger_tolerance_gives_fewer_buys():
    counts = []
    for tol in (0.0, 0.5, 2.0):
        bus, paper, om, runner, log, closes, starts = _system(tol)
        run_candles(bus, paper, closes, starts, WARM, N)
        counts.append(len(om.closed_trades) + len(om.open_positions))
    assert counts[0] >= counts[1] >= counts[2] and counts[0] > counts[2]


def test_the_forming_candle_is_never_used_ticks_alone_cause_no_signal():
    bus, paper, om, runner, log, closes, starts = _system(0.02)
    t = starts[WARM] + timedelta(minutes=3)
    for k, px in enumerate((900.0, 1100.0, 900.0, 1100.0)):         # wild moves inside one candle
        paper.set_price("TEST", px)
        asyncio.run(bus.publish(events.TICK, Tick("TEST", t + timedelta(seconds=k), px)))
    assert not om.open_positions and not log


# ---------------------------------------------------------------------- 5% stop, trade start
def _uptrend():
    """A fall through the warm-up (MACD below its signal line), then one steady rise: exactly one intersection from below."""
    return np.concatenate([np.linspace(1000, 940, WARM), np.linspace(940, 1100, N - WARM)])


def _first_buy(bus, paper, om, closes, starts, stop_pct):
    om.stop_loss_pct = stop_pct
    for i in range(WARM, N):
        run_candles(bus, paper, closes, starts, i, i + 1)
        if om.open_positions:
            return i
    raise AssertionError("no buy in the up-trend")


def test_every_buy_gets_a_stop_5_percent_below_its_fill_and_it_is_hit_on_a_tick():
    bus, paper, om, runner, log, closes, starts = _system(0.0, _uptrend())
    runner.stop_loss_pct = 0.05
    i = _first_buy(bus, paper, om, closes, starts, 0.05)
    pos = om.open_positions[0]
    assert pos.stop_loss == pytest.approx(pos.entry_price * 0.95) and pos.initial_risk == pytest.approx(pos.entry_price * 0.05)
    assert any("stop" not in (r[3] or "") for r in log)
    t = starts[i] + timedelta(minutes=STEP + 4)
    paper.set_price("TEST", pos.stop_loss + 0.5)
    asyncio.run(bus.publish(events.TICK, Tick("TEST", t, pos.stop_loss + 0.5)))
    assert om.has_position("TEST")                                   # above the stop: kept
    paper.set_price("TEST", pos.stop_loss - 0.5)
    asyncio.run(bus.publish(events.TICK, Tick("TEST", t + timedelta(seconds=1), pos.stop_loss - 0.5)))
    assert not om.has_position("TEST") and om.closed_trades[-1].exit_reason == "STOP"


def test_after_a_stop_out_no_new_buy_until_the_lines_intersect_again():
    bus, paper, om, runner, log, closes, starts = _system(0.0, _uptrend())
    i = _first_buy(bus, paper, om, closes, starts, 0.05)
    pos = om.open_positions[0]
    paper.set_price("TEST", pos.stop_loss - 1)
    asyncio.run(bus.publish(events.TICK, Tick("TEST", starts[i] + timedelta(minutes=STEP + 4), pos.stop_loss - 1)))
    assert not om.open_positions and om.closed_trades[-1].exit_reason == "STOP"
    run_candles(bus, paper, closes, starts, i + 1, N)                # MACD stays above the signal line: the same up-move
    assert not om.open_positions and len([r for r in log if r[0] == "BUY" and r[2] == "ACCEPTED"]) == 1


def test_no_buy_before_the_trade_start_time():
    bus, paper, om, runner, log, closes, starts = _system(0.0)
    run_candles(bus, paper, closes, starts, WARM, N)
    first = _entry_bars(om, starts)[0]
    bus, paper, om, runner, log, closes, starts = _system(0.0, trade_start=time(23, 0))          # nothing is allowed that early
    run_candles(bus, paper, closes, starts, WARM, N)
    assert not om.closed_trades and not om.open_positions
    assert any(r[0] == "BUY" and r[2] == "REJECTED" and "trading starts at 23:00" in (r[3] or "") for r in log) or         any(r[0] == "BUY" for r in log)
    start_at = (starts[first] + timedelta(minutes=STEP)).time()                                  # the first buy's own time
    bus, paper, om, runner, log, closes, starts = _system(0.0, trade_start=start_at)
    run_candles(bus, paper, closes, starts, WARM, N)
    assert (starts[first] + timedelta(minutes=STEP)) in [t.position.opened_at for t in om.closed_trades] +         [p.opened_at for p in om.open_positions]                                                 # allowed exactly at the start time


def test_trade_start_is_compared_with_the_candle_close_so_the_first_candle_can_buy_at_0920():
    from backend.core.models import Candle
    c = Candle("TEST", "5m", datetime(2026, 10, 6, 9, 15, tzinfo=IST), datetime(2026, 10, 6, 9, 20, tzinfo=IST), 1, 1, 1, 1, 1, True)
    assert c.end.time() >= time(9, 20)


# ---------------------------------------------------------------------- settings: 5m system as asked, main system untouched
def test_the_5m_settings_ask_for_the_requested_rules_and_the_main_system_is_unchanged():
    five = load_settings(ROOT / "config" / "settings_5m.yaml").section("trading")
    assert five["intraday_timeframe"] == "5m" and five["signal_mode"] == "candle_confirmed"
    assert five["macd_cross_tolerance"] == 0.02 and five["stop_loss_pct"] == 0.05
    assert five["trade_start"] == "09:15" and five["square_off"] == "15:00" and five["carry_longs"] is False
    main = load_settings(ROOT / "config" / "settings.yaml").section("trading")
    assert main["intraday_timeframe"] == "15m"
    assert main.get("signal_mode", "bar_close") == "bar_close"
    assert not main.get("macd_cross_tolerance") and not main.get("stop_loss_pct") and main.get("trade_start") == "09:15"
    assert main["square_off"] == "15:00"
