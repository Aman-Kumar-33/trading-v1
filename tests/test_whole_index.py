"""The whole NIFTY 50 is monitored and traded by both systems (no top-10 list), with no limit on open positions; the chart draws the
system's own closes; both_systems.exe opens the two systems in two browser tabs.

Run from second-system/:  python -m pytest tests -q
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_auto_mode import CAP, _fill_other_positions  # noqa: E402
from test_macd_system import PER_DAY, build_system, make_closes, run_candles, session_starts  # noqa: E402
from test_next_session import NEXT_DAY, TODAY, UNIVERSE, _save  # noqa: E402

from backend.analytics import dashboard_data as dd  # noqa: E402
from backend.core.config import load_settings  # noqa: E402
from backend.data import chart_history as ch, next_session as ns, trending, watchlist  # noqa: E402
from backend.execution import sizing  # noqa: E402
from backend.notify import telegram as tg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setattr(trending, "universe_symbols", lambda _s: UNIVERSE)
    sent = []
    monkeypatch.setattr(tg, "notify", lambda text, *a, **k: sent.append(text))
    return sent


def _pair(tmp_path):
    """The real settings files, only the folders moved into tmp_path - exactly as the two systems are configured."""
    main = tmp_path / "main.yaml"
    main.write_text(f"extends: {(ROOT / 'config' / 'settings.yaml').as_posix()}\ninstance:\n  history_peers: []\n"
                    f"paths:\n  live_data: {(tmp_path / 'data').as_posix()}\n  logs: {(tmp_path / 'logs').as_posix()}\n", encoding="utf-8")
    five = tmp_path / "five.yaml"
    five.write_text(f"extends: {(ROOT / 'config' / 'settings_5m.yaml').as_posix()}\ninstance:\n  history_peers: []\n  follow: {main.as_posix()}\n"
                    f"paths:\n  live_data: {(tmp_path / 'data_five').as_posix()}\n  logs: {(tmp_path / 'logs').as_posix()}\n", encoding="utf-8")
    main.write_text(main.read_text(encoding="utf-8").replace("history_peers: []", f"history_peers: [{five.as_posix()}]"), encoding="utf-8")   # as in the real config
    return load_settings(main), load_settings(five)


# ---------------------------------------------------------------------- the real config
def test_both_real_configs_trade_the_whole_index_with_at_most_15_open_positions():
    for name in ("settings.yaml", "settings_5m.yaml"):
        t = load_settings(ROOT / "config" / name).section("trading")
        assert t["trade_universe"] == "all" and t["max_open_positions"] == 15 and t["trade_start"] == "09:15" and t["square_off"] == "15:00"
        assert t["max_concurrent_positions"] == 15


# ---------------------------------------------------------------------- the stock list
def test_every_stock_is_active_for_both_systems_and_nothing_is_ranked(tmp_path):
    main, five = _pair(tmp_path)
    assert watchlist.trades_whole_index(main) and watchlist.trades_whole_index(five)
    syms, source = watchlist.active_symbols(main, "paper")
    assert syms == UNIVERSE and "whole" in source.lower()
    assert watchlist.active_symbols(five, "paper")[0] == UNIVERSE                       # the 5 minute system trades the same stocks
    assert watchlist.is_auto_mode(main, "paper") and not watchlist.ranks_stocks(main, "paper")


def test_a_hand_given_list_still_wins_over_the_whole_index(tmp_path):
    main, five = _pair(tmp_path)
    watchlist.save(main, "paper", ["S01", "S02"])
    assert watchlist.active_symbols(main, "paper")[0] == ["S01", "S02"]
    assert watchlist.active_symbols(five, "paper")[0] == ["S01", "S02"]                 # and the 5 minute system follows it
    assert not watchlist.is_auto_mode(main, "paper")


def test_the_dashboard_says_the_whole_index_is_traded(tmp_path):
    main, five = _pair(tmp_path)
    v = dd.trending_view(main, "paper")
    assert v["whole_index"] and v["stocks_now"] == UNIVERSE and v["picks"] == []
    v5 = dd.trending_view(five, "paper")
    assert v5["whole_index"] and v5["follows"] and v5["stocks_now"] == UNIVERSE
    cfg = dd.trading_config(main)
    assert cfg["whole_index"] is True and cfg["max_open_positions"] == 15
    assert dd.chart_context(five, "paper", "S05")["listed"] is True                       # every stock's chart: "on the list"


# ---------------------------------------------------------------------- after the close
def test_after_the_close_the_histories_of_all_stocks_are_saved_and_both_systems_get_the_same_list(tmp_path, _quiet):
    main, five = _pair(tmp_path)
    _save(main, "15m", 15)
    _save(five, "5m", 5)
    assert ns.after_close(five, TODAY) is True and ns.after_close(main, TODAY) is True      # no waiting for a ranking
    for system, tf in ((main, "15m"), (five, "5m")):
        m = json.loads((system.paths.live_data / "state" / "next_session.json").read_text(encoding="utf-8"))
        assert m["ready"] and m["list"] == UNIVERSE and m["for_session"] == NEXT_DAY.isoformat() and m["timeframe"] == tf
        assert len(m["histories"]) == len(UNIVERSE) and all(h["complete"] and h["bars"] > 100 for h in m["histories"].values())
    assert ns.priority_symbols(main, TODAY) == UNIVERSE == ns.priority_symbols(five, TODAY)
    assert trending.load_next(main, "paper", NEXT_DAY) is None                                # no top-N file is written
    texts = [t for t in _quiet if "HISTORY READY" in t]
    assert len(texts) == 2 and all("whole index" in t and f"all {len(UNIVERSE)}" in t for t in texts)


def test_not_ready_while_the_history_of_too_many_stocks_is_missing(tmp_path):
    main, five = _pair(tmp_path)
    _save(main, "15m", 15, UNIVERSE[:5])
    assert ns.after_close(main, TODAY) is False
    _save(main, "15m", 15, UNIVERSE[:11])                                                      # 11 of 12 = 92% >= 90%
    assert ns.after_close(main, TODAY) is False                                                # ...but the manifest is ready only when ALL are complete
    m = json.loads((main.paths.live_data / "state" / "next_session.json").read_text(encoding="utf-8"))
    assert m["ready"] is False and sum(h["complete"] for h in m["histories"].values()) == 11


# ---------------------------------------------------------------------- no limit on open positions
def test_plan_size_has_no_slot_limit_when_max_open_is_zero():
    p = sizing.plan_size(1000.0, CAP, 300_000.0, 20, 0, 0.25, 0.05, 0.25)
    assert p.qty == 250                                                                         # nothing is held back for "slots still open"
    p = sizing.plan_size(1000.0, CAP, 30_000.0, 20, 0, 0.25, 0.05, 0.25)
    assert p.qty == 0 and "not enough free cash" in p.skip_reason                               # the free cash is the only limit


def test_many_positions_do_not_block_a_buy(_quiet):
    n = 300
    closes = make_closes(n)
    starts = session_starts(n // PER_DAY + 1)[:n]
    bus, paper, om, runner, log = build_system(closes, starts, 100)
    assert runner.max_open_positions == 0
    _fill_other_positions(om, 25)                                                               # far more than the old limit of 10
    run_candles(bus, paper, closes, starts, 100, n)
    assert om.has_position("TEST") or om.closed_trades
    assert not any("max open positions" in str(why) for _, _, _, why in log)
    assert any("positions open)" in m and "/0" not in m for m in _quiet)


# ---------------------------------------------------------------------- the chart draws the system's own closes
def test_chart_uses_the_systems_own_closes_where_it_recorded_bars_and_leaves_the_rest(tmp_path):
    ts = ["2026-11-18 09:15:00+05:30", "2026-11-18 09:20:00+05:30", "2026-11-18 09:25:00+05:30", "2026-11-18 15:15:00+05:30"]
    out = pd.DataFrame({"ts": ts, "open": [100.0] * 4, "high": [101.0] * 4, "low": [99.0] * 4, "close": [100.5, 100.5, 100.5, 100.5], "volume": [1] * 4})
    own = pd.DataFrame({"ts": ["2026-11-18T09:20:00+05:30", "2026-11-18T09:25:00+05:30", "2026-11-18T15:15:00+05:30"],
                        "open": [100.0] * 3, "high": [103.0] * 3, "low": [98.0] * 3, "close": [101.5, 98.5, 55.0], "volume": [1] * 3})
    res = dd.with_own_closes(out, own)
    assert list(res["close"]) == [100.5, 101.5, 98.5, 100.5]          # bar 1 has no own record; the 15:15 closing bar is never replaced
    assert res["high"].iloc[1] == 101.5 and res["low"].iloc[2] == 98.5      # a candle always contains its close
    assert list(out["close"]) == [100.5] * 4                              # the input is not changed
    assert dd.with_own_closes(out, own.iloc[0:0]) is out


# ---------------------------------------------------------------------- both_systems.exe: two tabs, one instance
def _launcher():
    spec = importlib.util.spec_from_file_location("both_systems_under_test", ROOT / "scripts" / "both_systems.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_launcher_opens_the_two_systems_in_two_tabs(monkeypatch):
    mod = _launcher()
    opened = []
    monkeypatch.setattr(mod.webbrowser, "open", lambda url, *a, **k: opened.append(("window", url)))
    monkeypatch.setattr(mod.webbrowser, "open_new_tab", lambda url, *a, **k: opened.append(("tab", url)))
    monkeypatch.setattr(mod.time, "sleep", lambda *_: None)
    mod.open_tabs()
    assert opened == [("window", "http://127.0.0.1:8766/"), ("tab", "http://127.0.0.1:8767/")]


def test_a_second_launch_only_opens_the_tabs(monkeypatch, tmp_path):
    mod = _launcher()
    monkeypatch.setattr(mod, "LOCK", tmp_path / "both_systems.lock")
    mod.LOCK.write_text(str(os.getppid()), encoding="utf-8")                                    # a live "supervisor" other than this process
    opened = []
    monkeypatch.setattr(mod, "open_tabs", lambda: opened.append("tabs"))
    monkeypatch.setattr(mod, "spawn", lambda key: pytest.fail("a second launcher must not start anything"))
    assert mod.another_running() is True
    assert mod.supervise(open_browser=True) == 0 and opened == ["tabs"]
    mod.LOCK.write_text("999999999", encoding="utf-8")                                           # a dead one: not "running"
    assert mod.another_running() is False


# ---------------------------------------------------------------------- at most 15 open positions
def test_a_16th_position_is_not_opened_and_the_15th_is(_quiet):
    n = 300
    closes = make_closes(n)
    starts = session_starts(n // PER_DAY + 1)[:n]
    bus, paper, om, runner, log = build_system(closes, starts, 100)
    runner.max_open_positions = 15
    _fill_other_positions(om, 15)
    run_candles(bus, paper, closes, starts, 100, n)
    assert not om.has_position("TEST") and not om.closed_trades
    assert any(why == "max open positions 15/15" for _, _, _, why in log)
    bus, paper, om, runner, log = build_system(closes, starts, 100)
    runner.max_open_positions = 15
    _fill_other_positions(om, 14)                                                               # one slot left: the buy is taken
    run_candles(bus, paper, closes, starts, 100, n)
    assert om.has_position("TEST") or om.closed_trades
    assert any("/15 positions open)" in m for m in _quiet)


# ---------------------------------------------------------------------- no sell arrow after the square-off
def _crossing_candles():
    """5m candles of one day: up, down, up, then a sharp fall - the MACD crosses down at 13:10 and again at 15:00 (a candle that closes at
    15:05, after the square-off), with a cross up at 14:20 in between."""
    import numpy as np
    idx = pd.date_range("2026-11-17 09:15", "2026-11-17 15:25", freq="5min", tz="Asia/Kolkata")
    close = np.full(len(idx), 500.0)
    for i in range(1, len(idx)):
        h = idx[i].hour * 60 + idx[i].minute
        step = 0.5 if 660 <= h < 780 else -0.8 if 780 <= h < 830 else 0.8 if 830 <= h < 890 else -2.0 if h >= 890 else 0.0
        close[i] = close[i - 1] + step
    return pd.DataFrame({"ts": [str(x) for x in idx], "open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000})


def test_sell_arrows_after_the_square_off_are_not_drawn_but_earlier_ones_and_all_buys_stay():
    candles = _crossing_candles()
    plain = dd.macd_overlay(candles, len(candles))
    hidden = dd.macd_overlay(candles, len(candles), square_off="15:00", tf_minutes=5)
    sells_all = [s["ts"][11:16] for s in plain["signals"] if s["side"] == "SELL"]
    sells_kept = [s["ts"][11:16] for s in hidden["signals"] if s["side"] == "SELL"]
    assert sells_all == ["13:10", "15:00"], sells_all
    assert sells_kept == ["13:10"]                                                              # the 15:00 candle closes at 15:05 - after the square-off
    assert [s for s in plain["signals"] if s["side"] == "BUY"] == [s for s in hidden["signals"] if s["side"] == "BUY"]


def test_the_square_off_rule_does_not_apply_to_long_timeframes():
    candles = _crossing_candles()
    assert dd.macd_overlay(candles, len(candles), square_off="15:00", tf_minutes=None)["signals"] == dd.macd_overlay(candles, len(candles))["signals"]


# ---------------------------------------------------------------------- the saved histories are temporary
def test_a_refresh_keeps_only_the_window_and_the_prune_deletes_older_days(tmp_path):
    from datetime import timedelta
    from backend.data import history_store
    main, _ = _pair(tmp_path)
    _save(main, "15m", 15, UNIVERSE[:2])                                                        # sessions 2-17 Nov
    path = history_store.history_path(main, "15m", UNIVERSE[0])
    full = history_store.load_history(main, UNIVERSE[0], "15m")
    assert full["ts"].dt.date.min().isoformat() == "2026-11-02"
    trimmed = history_store.trim_to_window(full, 7, TODAY)                                        # 7 calendar days: from 10 Nov (a holiday), so 11 Nov
    assert trimmed["ts"].dt.date.min().isoformat() == "2026-11-11" and trimmed["ts"].iloc[-1] == full["ts"].iloc[-1]
    # the prune uses the real windows (60 days for 15m): nothing to cut here...
    assert ch.prune_history(main, TODAY) == 0
    # ...but a file with bars from long ago is shortened and the new days stay
    old = full.copy()
    old["ts"] = old["ts"] - timedelta(days=100)
    pd.concat([old, full]).to_parquet(path)
    assert ch.prune_history(main, TODAY) == 1
    assert history_store.load_history(main, UNIVERSE[0], "15m")["ts"].dt.date.min().isoformat() == "2026-11-02"


# ---------------------------------------------------------------------- both timeframes must be saved before it says ready
def test_ready_needs_both_5m_and_15m_of_every_stock(tmp_path, _quiet):
    main, five = _pair(tmp_path)
    _save(main, "15m", 15)
    assert ns.after_close(main, TODAY) is False                                                 # 15m is saved, the 5m series is not
    m = json.loads((main.paths.live_data / "state" / "next_session.json").read_text(encoding="utf-8"))
    assert m["timeframes"]["15m"] == {"complete": len(UNIVERSE), "of": len(UNIVERSE)} and m["timeframes"]["5m"]["complete"] == 0 and not m["ready"]
    assert not [t for t in _quiet if "HISTORY READY" in t]
    _save(five, "5m", 5)
    assert ns.after_close(main, TODAY) is True and ns.after_close(five, TODAY) is True
    m = json.loads((five.paths.live_data / "state" / "next_session.json").read_text(encoding="utf-8"))
    assert m["ready"] and m["timeframes"]["5m"]["complete"] == m["timeframes"]["15m"]["complete"] == len(UNIVERSE)
    assert all("5m and 15m" in t for t in _quiet if "HISTORY READY" in t)
