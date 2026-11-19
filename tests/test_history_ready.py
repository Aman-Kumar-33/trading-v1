"""History loaded after the close, ready for the next morning; both systems start by themselves at 09:15.

Run from second-system/:  python -m pytest tests -q
"""
from __future__ import annotations

import asyncio
import sys
import time as _t
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_chart_history import DAYS, ROOT, _bars, _settings  # noqa: E402

from backend.core.clock import IST  # noqa: E402
from backend.core.config import load_settings  # noqa: E402
from backend.data import chart_history as ch, history_store  # noqa: E402

KEYS = ["ts", "open", "high", "low", "close", "volume"]


def _tf(step: int) -> str:
    return "1h" if step == 60 else f"{step}m"


def _closing_day(day, step, real: bool):
    """A day of bars; the closing bars are either real (as Angel serves them right after 15:30) or already settled."""
    b = _bars([day], step)
    if not real:
        mod = b["ts"].dt.hour * 60 + b["ts"].dt.minute
        closing = b[mod >= 15 * 60 + 15]
        b = b[mod < 15 * 60 + 15]
        last = closing.iloc[-1]
        flat = {"ts": last["ts"], "open": last["close"], "high": last["close"], "low": last["close"], "close": last["close"], "volume": 0}
        b = pd.concat([b, pd.DataFrame([flat])], ignore_index=True)
    return b


@pytest.mark.parametrize("step", [5, 15, 30, 60])
def test_a_series_pulled_right_after_the_close_equals_the_one_pulled_next_morning(step):
    evening = history_store.settle_closing_bars(_closing_day(DAYS[0], step, real=True), _tf(step))
    morning = _closing_day(DAYS[0], step, real=False)
    assert evening[KEYS].reset_index(drop=True).equals(morning[KEYS].reset_index(drop=True))
    again = history_store.settle_closing_bars(morning, _tf(step))
    assert again[KEYS].reset_index(drop=True).equals(morning[KEYS].reset_index(drop=True))   # a settled day is left alone


def test_settling_collapses_the_closing_bars_of_each_day_into_one_flat_bar_at_the_official_close():
    b = _bars(DAYS[:2], 5)
    out = history_store.settle_closing_bars(b, "5m")
    for d in DAYS[:2]:
        day = out[out["ts"].dt.date == d.date()]
        tail = day[day["ts"].dt.hour * 60 + day["ts"].dt.minute >= 15 * 60 + 15]
        assert len(tail) == 1 and tail["ts"].iloc[0].strftime("%H:%M") == "15:25" and tail["volume"].iloc[0] == 0
        assert tail["open"].iloc[0] == tail["high"].iloc[0] == tail["low"].iloc[0] == tail["close"].iloc[0]
        assert tail["close"].iloc[0] == b[b["ts"].dt.date == d.date()]["close"].iloc[-1]
    assert len(out) == len(b) - 2 * 2                                                         # 15:15 and 15:20 gone, on each day
    daily = _bars(DAYS[:1], 15)
    assert history_store.settle_closing_bars(daily, "1d") is daily                            # daily bars are never touched


def test_a_series_covers_a_session_only_down_to_its_closing_bar():
    b5 = _bars(DAYS[:1], 5)
    assert history_store.covers_session(b5, "5m", DAYS[0].date())
    assert not history_store.covers_session(b5[b5["ts"] < datetime(2026, 11, 11, 15, 5, tzinfo=IST)], "5m", DAYS[0].date())   # pulled at 15:05
    assert history_store.covers_session(_bars(DAYS[:2], 5), "5m", DAYS[0].date())             # a later day is in: this one is complete
    assert not history_store.covers_session(None, "5m", DAYS[0].date())
    assert history_store.closing_bar_start("15m").strftime("%H:%M") == "15:15"
    assert history_store.closing_bar_start("5m").strftime("%H:%M") == "15:25"


def test_fetch_history_settles_the_closing_bars():
    class FakeAngel:
        def get_candle_data(self, info, tf, start, end):
            return _closing_day(DAYS[0], 5, real=True), None
    out = history_store.fetch_history(FakeAngel(), {}, "5m", 5, datetime(2026, 11, 12, 9, 0, tzinfo=IST))
    assert out["ts"].dt.strftime("%H:%M").tolist()[-2:] == ["15:10", "15:25"] and out["volume"].iloc[-1] == 0


def test_a_start_before_the_first_bar_uses_the_series_pulled_after_yesterdays_close(tmp_path):
    s, _ = _settings(tmp_path)
    full = _bars(DAYS, 5)                                                                   # holds 17 Nov's closing bar
    at = lambda h, m, sec=0: datetime(2026, 11, 18, h, m, sec, tzinfo=IST)                  # noqa: E731
    assert ch.usable_at_start(s, full, "5m", at(9, 15, 20))
    assert ch.usable_at_start(s, _bars(DAYS, 15), "15m", at(9, 25))                         # the 15m first bar closes at 09:30
    assert not ch.usable_at_start(s, full, "5m", at(11, 0))                                 # a later start needs today's bars too
    cut = full[full["ts"] < datetime(2026, 11, 17, 15, 5, tzinfo=IST)]                      # pulled at 15:05: closing bars missing
    assert not ch.usable_at_start(s, cut, "5m", at(9, 15, 20))
    assert not ch.usable_at_start(s, None, "5m", at(9, 15, 20))


def test_loading_from_disk_does_not_wait_between_stocks():
    syms = [f"S{i}" for i in range(20)]
    done = []

    async def run():
        stop, lock = asyncio.Event(), asyncio.Lock()

        async def fetch_one(sym):
            df = _bars(DAYS[:1], 15)
            df.attrs["from_cache"] = True
            return df
        return await history_store.load_all(syms, fetch_one, lambda s_, df: done.append(s_), stop, lock, spacing=1.0)

    t = _t.time()
    assert asyncio.run(run()) == [] and len(done) == 20 and _t.time() - t < 5               # 20 x 1 s of spacing would take 20 s


def test_after_the_close_each_system_says_once_that_tomorrow_is_ready(tmp_path, monkeypatch):
    from backend.notify import telegram as tg
    s, _ = _settings(tmp_path)
    sent = []
    monkeypatch.setattr(tg, "notify", lambda text, *a, **k: sent.append(text))
    now = datetime(2026, 11, 18, 15, 40, tzinfo=IST)
    assert ch.announce_ready(s, now) is True and ch.announce_ready(s, now + timedelta(minutes=10)) is False
    assert len(sent) == 1 and "HISTORY READY" in sent[0] and "Thu 19 Nov" in sent[0]
    assert ch.announce_ready(s, now + timedelta(days=1)) is True                            # a new day: told again


def test_the_warmer_waits_for_the_closing_bar_of_the_day_before_calling_a_stock_ready(tmp_path, monkeypatch):
    from backend.data import trending
    s, _ = _settings(tmp_path)
    monkeypatch.setattr(trending, "universe_symbols", lambda _s: ["AAA"])
    after = datetime(2026, 11, 17, 15, 40, tzinfo=IST)
    partial = _bars(DAYS, 15)
    partial = partial[partial["ts"] < datetime(2026, 11, 17, 15, 15, tzinfo=IST)]            # pulled before the closing bar existed
    path = history_store.history_path(s, "15m", "AAA")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial.to_parquet(path)
    asked = []

    def fetch(settings, symbol, tf, now):
        asked.append(symbol)
        _bars(DAYS, 15).to_parquet(path)
        return True
    assert ch.warm_once(s, now=after, fetch=fetch) == [] and asked == ["AAA"]
    asked.clear()
    assert ch.warm_once(s, now=after, fetch=fetch) == [] and asked == []                    # now complete: left alone


def test_the_trading_process_starts_at_the_auto_start_time_once_a_day_and_never_after_the_square_off(tmp_path, monkeypatch):
    from backend.analytics import dashboard_data
    from backend.execution import process_control as pc
    s, _ = _settings(tmp_path)
    assert s.section("trading")["auto_start_time"] == "09:10"
    started = []
    monkeypatch.setattr(pc, "start", lambda settings, mode, *a, **k: started.append(mode) or {"running": True})
    monkeypatch.setattr(pc, "status", lambda *a, **k: {"running": False})

    def at(hh, mm, state="OPEN"):
        monkeypatch.setattr(dashboard_data, "market_status",
                            lambda _s: {"state": state, "is_open": state == "OPEN", "ts": f"2026-11-18T{hh:02d}:{mm:02d}:05+05:30"})
        pc.maybe_auto_start_paper(s)

    at(9, 5, "PRE_OPEN")
    at(9, 9, "PRE_OPEN")
    assert started == []                                                                    # before 09:10: nothing
    at(9, 10, "PRE_OPEN")
    at(9, 16)
    assert started == ["paper"]                                                             # at 09:10 (listening before the open), once
    started.clear()
    (tmp_path / "other").mkdir()
    s2, _ = _settings(tmp_path / "other")
    monkeypatch.setattr(dashboard_data, "market_status",
                        lambda _s: {"state": "SQUARE_OFF", "is_open": True, "ts": "2026-11-18T15:10:05+05:30"})
    pc.maybe_auto_start_paper(s2)
    assert started == []                                                                    # after the 15:00 square-off: never


def test_both_systems_are_set_to_auto_start_and_square_off_at_three():
    five = load_settings(ROOT / "config" / "settings_5m.yaml")
    main = load_settings(ROOT / "config" / "settings.yaml")
    for cfg in (main, five):
        t = cfg.section("trading")
        assert t["auto_start_time"] == "09:10" and t["trade_start"] == "09:15" and t["square_off"] == "15:00" and t["carry_longs"] is False
    assert five.section("instance")["auto_start"] is True and main.section("instance")["auto_start"] is True
