"""Dashboard chart history: fresh Angel series only - never the old data lake glued to live bars.

Run from second-system/:  python -m pytest tests -q
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.analytics import dashboard_data as dd  # noqa: E402
from backend.core.clock import IST  # noqa: E402
from backend.core.config import load_settings  # noqa: E402
from backend.data import chart_history as ch, history_store  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DAYS = [datetime(2026, 11, d, tzinfo=IST) for d in (11, 12, 13, 16, 17)]      # trading days (10 Nov is a holiday)
NOW = datetime(2026, 11, 18, 9, 20, tzinfo=IST)                               # the morning after 17 Nov's close, no bar closed yet
NOW_LATER = datetime(2026, 11, 18, 9, 46, tzinfo=IST)                         # the 09:15 and 09:30 15m bars have closed


def _bars(days, step: int, stub: bool = True, base=1000.0) -> pd.DataFrame:
    rows, px = [], base
    for d in days:
        t = d.replace(hour=9, minute=15)
        end = d.replace(hour=15, minute=30)
        while t < end:
            px += 1.0
            rows.append((t, px - 0.5, px + 1, px - 1, px, 1000))
            t += timedelta(minutes=step)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    return df


def _settings(tmp_path, name="main", follow: Path | None = None):
    p = tmp_path / f"{name}.yaml"
    extra = (f"instance:\n  name: {name}\n  history_peers: []\n" + (f"  follow: {follow.as_posix()}\n" if follow else "")
             + "trading:\n  trade_universe: ranked\n")        # the top-N list is what most of these tests are about; test_whole_index.py covers "all"
    p.write_text(f"extends: {(ROOT / 'config' / 'settings.yaml').as_posix()}\n{extra}"
                 f"paths:\n  live_data: {(tmp_path / ('data_' + name)).as_posix()}\n  logs: {(tmp_path / 'logs').as_posix()}\n",
                 encoding="utf-8")
    return load_settings(p), p


def _cache(settings, symbol, tf, df):
    path = history_store.history_path(settings, tf, symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def _no_fetch(*_a, **_k):
    raise AssertionError("must not fetch from Angel here")


# ---------------------------------------------------------------------- resampling
def test_resample_aligns_to_the_open_and_keeps_the_closing_stub_as_its_own_bar():
    b15 = _bars(DAYS[:1], 15)
    assert b15["ts"].iloc[-1].strftime("%H:%M") == "15:15"                           # the closing 15:15-15:30 stub
    h1 = ch.resample(b15, 60)
    assert [t.strftime("%H:%M") for t in h1["ts"]] == ["09:15", "10:15", "11:15", "12:15", "13:15", "14:15", "15:15"]
    first = b15.iloc[:4]
    assert h1["open"].iloc[0] == first["open"].iloc[0] and h1["close"].iloc[0] == first["close"].iloc[-1]
    assert h1["high"].iloc[0] == first["high"].max() and h1["low"].iloc[0] == first["low"].min()
    assert h1["volume"].iloc[0] == first["volume"].sum()
    m30 = ch.resample(b15, 30)
    assert [t.strftime("%H:%M") for t in m30["ts"]][-2:] == ["14:45", "15:15"]


def test_a_half_built_bucket_is_left_out():
    b15 = _bars(DAYS[:1], 15).iloc[:3]                       # 09:15, 09:30, 09:45 -> the 30m bucket 09:45 is half built
    out = ch.resample(b15, 30)
    assert [t.strftime("%H:%M") for t in out["ts"]] == ["09:15"]


# ---------------------------------------------------------------------- where a chart's series comes from
def test_a_fresh_cache_of_the_timeframe_is_used_as_is(tmp_path):
    s, _ = _settings(tmp_path)
    _cache(s, "SBIN", "15m", _bars(DAYS, 15))
    out = ch.chart_history(s, "SBIN", "15m", 100, now=NOW, fetch=_no_fetch)
    assert len(out) == 100 and out["ts"].iloc[-1].date() == DAYS[-1].date()


def test_longer_timeframes_are_built_from_the_fresh_cache_without_a_gap_and_without_fetching(tmp_path):
    s, _ = _settings(tmp_path)
    _cache(s, "SBIN", "15m", _bars(DAYS, 15))
    for tf in ("30m", "1h"):
        out = ch.chart_history(s, "SBIN", tf, 1000, now=NOW, fetch=_no_fetch)
        days = sorted({t.date() for t in out["ts"]})
        assert days == [d.date() for d in DAYS]                          # every session, no hole between them
        assert out["ts"].is_monotonic_increasing and not out["ts"].duplicated().any()


def test_the_5_minute_system_reads_the_followed_systems_cache(tmp_path):
    main, main_path = _settings(tmp_path, "main")
    five, _ = _settings(tmp_path, "five", follow=main_path)
    _cache(main, "SBIN", "15m", _bars(DAYS, 15))
    out = ch.chart_history(five, "SBIN", "30m", 1000, now=NOW, fetch=_no_fetch)
    assert out is not None and out["ts"].iloc[-1].date() == DAYS[-1].date()


def test_a_timeframe_with_no_cache_is_fetched_once_from_angel(tmp_path):
    s, _ = _settings(tmp_path)
    calls = []

    def fetch(settings, symbol, tf, now):
        calls.append((symbol, tf))
        return _bars(DAYS, 5)

    out = ch.chart_history(s, "SBIN", "5m", 50, now=NOW, fetch=fetch)
    assert calls == [("SBIN", "5m")] and len(out) == 50


def test_a_stale_cache_is_replaced_by_a_fetch_and_kept_if_the_fetch_fails(tmp_path):
    s, _ = _settings(tmp_path)
    stale = _bars(DAYS[:2], 5)                                            # ends 12 Nov
    _cache(s, "SBIN", "5m", stale)
    fresh = ch.chart_history(s, "SBIN", "5m", 5000, now=NOW, fetch=lambda *a: _bars(DAYS, 5))
    assert fresh["ts"].iloc[-1].date() == DAYS[-1].date()
    failed = ch.chart_history(s, "SBIN", "5m", 5000, now=NOW, fetch=lambda *a: None)     # Angel refused: still draw something
    assert failed["ts"].iloc[-1].date() == DAYS[1].date()


def test_nothing_cached_and_nothing_fetched_gives_no_chart_not_a_wrong_one(tmp_path):
    s, _ = _settings(tmp_path)
    assert ch.chart_history(s, "SBIN", "1m", 50, now=NOW, fetch=lambda *a: None) is None


def test_todays_closed_live_bars_complete_the_newest_longer_bar_but_never_a_half_one(tmp_path):
    s, _ = _settings(tmp_path)
    _cache(s, "SBIN", "15m", _bars(DAYS, 15))
    today = datetime(2026, 11, 18, 9, 15, tzinfo=IST)

    def live(tf, after):
        assert tf == "15m"
        return pd.DataFrame({"ts": [today.isoformat(), (today + timedelta(minutes=15)).isoformat()],
                             "open": [1, 2], "high": [3, 3], "low": [0, 1], "close": [2, 3], "volume": [10, 10]})
    out = ch.chart_history(s, "SBIN", "30m", 5000, live=live, now=NOW_LATER, fetch=_no_fetch)
    assert out["ts"].iloc[-1] == today and out["close"].iloc[-1] == 3
    half = lambda tf, after: live(tf, after).iloc[:1]                       # noqa: E731 - only the 09:15 bar has closed
    out = ch.chart_history(s, "SBIN", "30m", 5000, live=half, now=NOW_LATER, fetch=_no_fetch)
    assert out["ts"].iloc[-1].date() == DAYS[-1].date()


# ---------------------------------------------------------------------- the dashboard route
def test_symbol_candles_never_uses_the_old_data_lake(tmp_path, monkeypatch):
    s, _ = _settings(tmp_path)
    monkeypatch.setattr(ch._fetcher, "fetch", lambda *a: None)                # never reach the real Angel API from a test
    _cache(s, "SBIN", "15m", _bars(DAYS, 15))
    lake = s.paths.clean / "30m"
    lake.mkdir(parents=True, exist_ok=True)
    old = _bars([datetime(2026, 9, 25, tzinfo=IST)], 30, base=500.0)
    old.to_parquet(lake / "SBIN.parquet")
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("create table candles (symbol, timeframe, ts_start, open, high, low, close, volume)")
    out = dd.symbol_candles(s, conn, "SBIN", "30m", 1000)
    assert not out.empty and pd.to_datetime(out["ts"], format="mixed", utc=True).dt.tz_convert("Asia/Kolkata").min().month == 11
    assert out["close"].min() > 900                                          # nothing from the 500-priced September file


def test_a_daily_bar_is_final_only_after_the_close():
    d = pd.DataFrame({"ts": pd.to_datetime(["2026-11-17", "2026-11-18"]).tz_localize("Asia/Kolkata"),
                      "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1})
    assert len(history_store.drop_forming(d, "1d", datetime(2026, 11, 18, 9, 46, tzinfo=IST))) == 1
    assert len(history_store.drop_forming(d, "1d", datetime(2026, 11, 18, 15, 31, tzinfo=IST))) == 2


def test_last_finished_session_skips_weekends_and_holidays(tmp_path):
    s, _ = _settings(tmp_path)
    assert ch.last_finished_session(s, datetime(2026, 11, 18, 9, 46, tzinfo=IST)) == DAYS[-1].date()          # before today's close
    assert ch.last_finished_session(s, datetime(2026, 11, 18, 16, 0, tzinfo=IST)).day == 18
    assert ch.last_finished_session(s, datetime(2026, 11, 14, 12, 0, tzinfo=IST)) == DAYS[2].date()           # Saturday -> Fri 13th


def test_with_no_history_old_stray_live_bars_are_not_shown_as_a_chart(tmp_path, monkeypatch):
    s, _ = _settings(tmp_path)
    monkeypatch.setattr(ch._fetcher, "fetch", lambda *a: None)
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("create table candles (symbol, timeframe, ts_start, open, high, low, close, volume)")
    conn.execute("insert into candles values ('SBIN','1d','2026-11-08T14:20:00+05:30',1,1,1,1,1)")        # a day the system ran, long ago
    last = ch.last_finished_session(s, datetime.now(IST)).isoformat()                                     # the latest finished session, whatever day it is now
    conn.execute(f"insert into candles values ('SBIN','1d','{last}T09:15:00+05:30',2,2,2,2,2)")
    out = dd.symbol_candles(s, conn, "SBIN", "1d", 100)
    assert list(out["close"]) == [2.0]                                                                      # only the latest session


def test_the_warmer_loads_every_missing_stock_and_reports_what_is_still_missing(tmp_path, monkeypatch):
    from backend.data import trending
    s, _ = _settings(tmp_path)
    monkeypatch.setattr(trending, "universe_symbols", lambda _s: ["AAA", "BBB", "CCC"])
    _cache(s, "AAA", "15m", _bars(DAYS, 15))                              # already fresh: not fetched
    asked = []

    def fetch(settings, symbol, tf, now):
        asked.append(symbol)
        if symbol == "BBB":                                               # Angel refuses this one
            return None
        _cache(settings, symbol, tf, _bars(DAYS, 15))
        return True

    missing = ch.warm_once(s, now=NOW, fetch=fetch)
    assert asked == ["BBB", "CCC"] and missing == ["BBB"]
    asked.clear()
    assert ch.warm_once(s, now=NOW, fetch=fetch) == ["BBB"] and asked == ["BBB"]      # next round retries only the missing one


# ---------------------------------------------------------------------- intraday freshness: the chart must not end in the past
def test_required_last_bar_is_the_newest_closed_bar_with_a_grace_period_and_stops_growing_at_the_end_of_data(tmp_path):
    s, _ = _settings(tmp_path)
    at = lambda h, m, sec=0: datetime(2026, 11, 18, h, m, sec, tzinfo=IST)           # noqa: E731
    assert ch.required_last_bar(s, at(9, 20), "5m") is None                         # nothing has closed yet (grace)
    assert ch.required_last_bar(s, at(10, 20, 30), "5m") == pd.Timestamp(at(10, 10))   # the 10:15 bar closed 30 s ago: not required yet
    assert ch.required_last_bar(s, at(10, 21, 40), "5m") == pd.Timestamp(at(10, 15))
    assert ch.required_last_bar(s, at(10, 21, 40), "15m") == pd.Timestamp(at(10, 0))
    assert ch.required_last_bar(s, at(15, 20), "5m") == pd.Timestamp(at(15, 10))        # Angel has no real bars after ~15:15
    assert ch.required_last_bar(s, at(16, 0), "5m") is None                             # after the close the date rule applies
    assert ch.required_last_bar(s, at(10, 20), "1d") is None
    assert ch.required_last_bar(s, datetime(2026, 11, 14, 10, 20, tzinfo=IST), "5m") is None   # a Saturday


def test_a_series_that_stops_at_09_50_is_refetched_at_10_21_and_not_again_within_a_minute(tmp_path):
    s, _ = _settings(tmp_path)
    old = _bars(DAYS, 5)
    old = old[old["ts"] < datetime(2026, 11, 17, 15, 30, tzinfo=IST)]
    today = pd.DataFrame({"ts": pd.date_range("2026-11-18 09:15", "2026-11-18 09:50", freq="5min", tz=IST), "open": 1.0, "high": 2.0,
                          "low": 0.5, "close": 1.5, "volume": 10})
    _cache(s, "SBIN", "5m", pd.concat([old, today], ignore_index=True))
    now = datetime(2026, 11, 18, 10, 21, 40, tzinfo=IST)
    asked = []

    def fetch(settings, symbol, tf, n):
        asked.append(tf)
        more = pd.DataFrame({"ts": pd.date_range("2026-11-18 09:55", "2026-11-18 10:15", freq="5min", tz=IST), "open": 1.0, "high": 2.0,
                             "low": 0.5, "close": 1.6, "volume": 10})
        return pd.concat([old, today, more], ignore_index=True)

    out = ch.chart_history(s, "SBIN", "5m", 50, now=now, fetch=fetch)
    assert asked == ["5m"] and str(out["ts"].iloc[-1])[:16] == "2026-11-18 10:15"
    # a series that already reaches the newest closed bar is not fetched again
    _cache(s, "SBIN", "5m", fetch(s, "SBIN", "5m", now))
    asked.clear()
    ch.chart_history(s, "SBIN", "5m", 50, now=now, fetch=fetch)
    assert asked == []


def test_bars_the_live_system_built_count_so_a_running_system_is_not_refetched(tmp_path):
    s, _ = _settings(tmp_path)
    _cache(s, "SBIN", "15m", _bars(DAYS, 15))
    now = datetime(2026, 11, 18, 10, 21, 40, tzinfo=IST)
    live = lambda tf, after: pd.DataFrame({"ts": ["2026-11-18T10:00:00+05:30"], "open": [1], "high": [2], "low": [0], "close": [1], "volume": [1]})  # noqa: E731
    out = ch.chart_history(s, "SBIN", "15m", 50, live=live, now=now, fetch=_no_fetch)
    assert out["ts"].iloc[-1].date() == DAYS[-1].date()                              # served from the cache, no fetch needed
    with pytest.raises(AssertionError):                                              # without live bars it does go to Angel
        ch.chart_history(s, "SBIN", "15m", 50, now=now, fetch=_no_fetch)


# ---------------------------------------------------------------------- the on-demand fetcher: people go first, and never wait long
def _fake_fetcher(monkeypatch, tmp_path):
    f = ch._Fetcher()
    monkeypatch.setattr(f, "_session", lambda settings: (object(), {"SBIN": {"token": "1", "exchange": "NSE"}}))
    monkeypatch.setattr(ch._time, "sleep", lambda *_: None)
    seen = {}

    def refresh(settings, angel, symbol, info, tf, days, now, retry_waits=None):
        seen["retry_waits"] = retry_waits
        return _bars(DAYS, 5)
    monkeypatch.setattr(ch.history_store, "refresh_symbol", refresh)
    return f, seen


def test_a_chart_request_retries_angel_once_quickly_and_the_background_sweep_uses_the_full_retries(tmp_path, monkeypatch):
    s, _ = _settings(tmp_path)
    f, seen = _fake_fetcher(monkeypatch, tmp_path)
    assert f.fetch(s, "SBIN", "5m", NOW) is not None and seen["retry_waits"] == (3,)
    f2, seen2 = _fake_fetcher(monkeypatch, tmp_path)
    assert f2.fetch(s, "SBIN", "5m", NOW, interactive=False) is not None and seen2["retry_waits"] is None


def test_a_chart_request_gives_up_waiting_for_a_busy_session_instead_of_hanging(tmp_path, monkeypatch):
    s, _ = _settings(tmp_path)
    f, _seen = _fake_fetcher(monkeypatch, tmp_path)
    monkeypatch.setattr(ch, "INTERACTIVE_WAIT_SECONDS", 0.05)
    with f._lock:                                                            # the sweep is in the middle of a request
        assert f.fetch(s, "SBIN", "5m", NOW) is None and f.waiting == 0


def test_a_rate_limit_refusal_does_not_throw_the_session_away(tmp_path, monkeypatch):
    s, _ = _settings(tmp_path)
    f, _seen = _fake_fetcher(monkeypatch, tmp_path)
    f._angel = object()
    monkeypatch.setattr(ch.history_store, "refresh_symbol",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Access denied because of exceeding access rate")))
    assert f.fetch(s, "SBIN", "5m", NOW) is None and f._angel is not None


# ---------------------------------------------------------------------- a chart never waits for Angel while it has something to draw
def _behind_cache(s):
    old = _bars(DAYS, 5)
    old = old[old["ts"] < datetime(2026, 11, 17, 15, 30, tzinfo=IST)]
    today = pd.DataFrame({"ts": pd.date_range("2026-11-18 09:15", "2026-11-18 09:50", freq="5min", tz=IST), "open": 1.0, "high": 2.0,
                          "low": 0.5, "close": 1.5, "volume": 10})
    _cache(s, "SBIN", "5m", pd.concat([old, today], ignore_index=True))


def test_a_series_that_is_behind_is_returned_at_once_flagged_stale_and_refreshed_in_the_background(tmp_path):
    s, _ = _settings(tmp_path)
    _behind_cache(s)
    now = datetime(2026, 11, 18, 10, 21, 40, tzinfo=IST)
    refreshed = []
    out, stale = ch.chart_history_ex(s, "SBIN", "5m", 50, now=now, fetch=_no_fetch, blocking=False,
                                     refresh=lambda *a: refreshed.append(a[1:3]))
    assert stale and refreshed == [("SBIN", "5m")] and str(out["ts"].iloc[-1])[:16] == "2026-11-18 09:50"


def test_a_current_series_is_not_stale_and_starts_no_refresh(tmp_path):
    s, _ = _settings(tmp_path)
    _cache(s, "SBIN", "15m", _bars(DAYS, 15))
    out, stale = ch.chart_history_ex(s, "SBIN", "15m", 50, now=NOW, fetch=_no_fetch, blocking=False, refresh=_no_fetch)
    assert not stale and len(out) == 50


def test_only_a_stock_with_nothing_cached_waits_for_its_first_fetch(tmp_path):
    s, _ = _settings(tmp_path)
    out, stale = ch.chart_history_ex(s, "SBIN", "5m", 50, now=NOW, fetch=lambda *a: _bars(DAYS, 5), blocking=False, refresh=_no_fetch)
    assert not stale and len(out) == 50


def test_the_route_reports_stale_and_never_calls_angel_itself(tmp_path, monkeypatch):
    s, _ = _settings(tmp_path)
    _behind_cache(s)
    calls = []
    monkeypatch.setattr(ch._fetcher, "refresh_in_background", lambda *a: calls.append(a[1:3]))
    monkeypatch.setattr(ch._fetcher, "fetch", _no_fetch)
    monkeypatch.setattr(ch, "datetime", type("D", (), {"now": staticmethod(lambda tz=None: datetime(2026, 11, 18, 10, 21, 40, tzinfo=IST))}))
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("create table candles (symbol, timeframe, ts_start, open, high, low, close, volume)")
    out = dd.symbol_candles(s, conn, "SBIN", "5m", 100)
    assert out.attrs["stale"] is True and calls == [("SBIN", "5m")] and not out.empty


def test_two_requests_for_the_same_stock_start_one_background_fetch(tmp_path, monkeypatch):
    import threading
    s, _ = _settings(tmp_path)
    f = ch._Fetcher()
    started, release, count = threading.Event(), threading.Event(), []

    def slow_fetch(settings, symbol, tf, now, interactive=True):
        count.append(symbol)
        started.set()
        release.wait(5)

    monkeypatch.setattr(f, "fetch", slow_fetch)
    f.refresh_in_background(s, "SBIN", "5m", NOW)
    assert started.wait(5)
    f.refresh_in_background(s, "SBIN", "5m", NOW)                      # still running: no second one
    release.set()
    for _ in range(100):
        if not f._inflight:
            break
        threading.Event().wait(0.02)
    assert count == ["SBIN"] and not f._inflight


def test_the_main_dashboard_reads_the_5_minute_systems_cache_instead_of_fetching_it_again(tmp_path):
    five, five_path = _settings(tmp_path, "five")
    main_path = tmp_path / "main.yaml"
    main_path.write_text(f"extends: {(ROOT / 'config' / 'settings.yaml').as_posix()}\n"
                         f"instance:\n  history_peers: [{five_path.as_posix()}]\n".replace("\n", "\n") +
                         f"paths:\n  live_data: {(tmp_path / 'data_main').as_posix()}\n  logs: {(tmp_path / 'logs').as_posix()}\n", encoding="utf-8")
    main = load_settings(main_path)
    _cache(five, "SBIN", "5m", _bars(DAYS, 5))
    out = ch.chart_history(main, "SBIN", "5m", 100, now=NOW, fetch=_no_fetch)
    assert len(out) == 100 and out["ts"].iloc[-1].date() == DAYS[-1].date()
    assert len(ch._systems(five)) == 1                                   # a system never lists itself as its own peer


def test_the_shipped_settings_make_the_main_dashboard_a_reader_of_the_5_minute_cache():
    main = load_settings(ROOT / "config" / "settings.yaml")
    assert main.section("instance")["history_peers"] == ["settings_5m.yaml"]
    assert main.section("trading")["intraday_timeframe"] == "15m"


def _db_with_bars(path, symbol, tf, stamps):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("create table candles (symbol, timeframe, ts_start, open, high, low, close, volume)")
    for t in stamps:
        con.execute("insert into candles values (?,?,?,?,?,?,?,?)", (symbol, tf, t, 1, 2, 0.5, 1.5, 10))
    con.commit()
    con.close()


def test_the_idle_system_gets_todays_bars_from_the_running_systems_database_without_asking_angel(tmp_path):
    main, main_path = _settings(tmp_path, "main")
    five_path = tmp_path / "five.yaml"
    five_path.write_text(f"extends: {(ROOT / 'config' / 'settings.yaml').as_posix()}\ninstance:\n  history_peers: [{main_path.as_posix()}]\n"
                         f"paths:\n  live_data: {(tmp_path / 'data_five').as_posix()}\n  logs: {(tmp_path / 'logs').as_posix()}\n", encoding="utf-8")
    five = load_settings(five_path)
    _cache(main, "SBIN", "15m", _bars(DAYS, 15))                                  # the 15m history the main system keeps
    _db_with_bars(main.paths.db_file, "SBIN", "15m",
                  [f"2026-11-18T{h:02d}:{m:02d}:00+05:30" for h, m in ((9, 15), (9, 30), (9, 45), (10, 0), (10, 15))])
    now = datetime(2026, 11, 18, 10, 31, 40, tzinfo=IST)
    out, stale = ch.chart_history_ex(five, "SBIN", "15m", 500, now=now, fetch=_no_fetch, refresh=_no_fetch, blocking=False,
                                     live=lambda tf, after: ch.live_rows(five, "SBIN", tf, after))
    assert not stale                                                              # reaches the newest closed bar through main's DB
    live = ch.live_rows(five, "SBIN", "15m", "2026-11-17 15:15:00+05:30")
    assert len(live) == 5 and live["ts"].is_monotonic_increasing
    out30, stale30 = ch.chart_history_ex(five, "SBIN", "30m", 500, now=now, fetch=_no_fetch, refresh=_no_fetch, blocking=False,
                                         live=lambda tf, after: ch.live_rows(five, "SBIN", tf, after))
    assert not stale30 and str(out30["ts"].iloc[-1])[:16] == "2026-11-18 09:45"   # the 10:15 bucket has only one of its two 15m bars: left out


def test_the_warmer_does_not_fetch_a_stock_whose_newest_bars_are_in_a_live_database(tmp_path, monkeypatch):
    from backend.data import trending
    s, _ = _settings(tmp_path)
    monkeypatch.setattr(trending, "universe_symbols", lambda _s: ["SBIN"])
    _cache(s, "SBIN", "15m", _bars(DAYS, 15))
    _db_with_bars(s.paths.db_file, "SBIN", "15m", ["2026-11-18T09:15:00+05:30", "2026-11-18T09:30:00+05:30", "2026-11-18T09:45:00+05:30"])
    now = datetime(2026, 11, 18, 10, 0, 40, tzinfo=IST)                           # newest closed 15m bar must start 09:45
    assert ch.warm_once(s, now=now, fetch=_no_fetch) == []


# ---------------------------------------------------------------------- the arrows on a chart are what the system would do
def _candles_df(n=400):
    rng = np.random.default_rng(3)
    close = 1000 + np.cumsum(rng.normal(0, 3, n))
    ts = pd.date_range("2026-10-01 09:15", periods=n, freq="5min", tz=IST)
    return pd.DataFrame({"ts": ts.astype(str), "open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1})


def test_with_a_tolerance_the_buy_arrows_are_the_confirmed_entries_and_sells_are_unchanged():
    c = _candles_df()
    plain = dd.macd_overlay(c, 300)
    zero = dd.macd_overlay(c, 300, (0.0, 0.0))
    wide = dd.macd_overlay(c, 300, (0.02, 0.15))
    sides = lambda o, side: [x["ts"] for x in o["signals"] if x["side"] == side]            # noqa: E731
    assert sides(zero, "BUY") == sides(plain, "BUY")                                      # no tolerance: every plain cross
    assert sides(wide, "SELL") == sides(plain, "SELL")                                    # the exit is the plain cross either way
    assert 0 < len(sides(wide, "BUY")) < len(sides(plain, "BUY"))                         # a real tolerance drops weak crosses
    for x in wide["signals"]:
        if x["side"] == "BUY":
            assert x["macd"] > x["signal"]                                                # and every kept BUY is above the line


def test_the_chart_says_whether_the_system_is_running_and_whether_the_stock_is_on_its_list(tmp_path, monkeypatch):
    from backend.data import watchlist
    from backend.execution import process_control as pc
    s, _ = _settings(tmp_path)
    monkeypatch.setattr(watchlist, "active_symbols", lambda *a, **k: (["SBIN"], "test"))
    monkeypatch.setattr(pc, "status", lambda *a, **k: {"running": False})
    ctx = dd.chart_context(s, "paper", "SBIN")
    assert ctx["running"] is False and ctx["listed"] is True
    assert dd.chart_context(s, "paper", "TCS")["listed"] is False
