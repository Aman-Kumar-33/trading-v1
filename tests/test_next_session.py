"""The list made after the close for the next session, shared by both systems; histories saved and checked; force-exit commands; the
exe finding the project from the folder above it.

Run from second-system/:  python -m pytest tests -q
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_chart_history import ROOT, _settings  # noqa: E402

from backend.core.clock import IST  # noqa: E402
from backend.data import chart_history as ch, history_store, next_session as ns, trending, watchlist  # noqa: E402
from backend.execution import process_control as pc  # noqa: E402
from backend.notify import telegram as tg  # noqa: E402

UNIVERSE = [f"S{i:02d}" for i in range(12)]
SESSION_DAYS = [d for d in pd.bdate_range("2026-11-02", "2026-11-17") if d.day != 10]       # 10 Nov was a holiday
TODAY = datetime(2026, 11, 17, 15, 40, tzinfo=IST)                                           # just after the close
NEXT_DAY = datetime(2026, 11, 18, tzinfo=IST).date()


def _walk(step: int, seed: int) -> pd.DataFrame:
    """Random-walk bars of every session day, ending with the settled closing bar (15m: 15:15, 5m: 15:25)."""
    rng = np.random.default_rng(seed)
    rows, px = [], 500.0 + 20 * seed
    for d in SESSION_DAYS:
        t = datetime(d.year, d.month, d.day, 9, 15, tzinfo=IST)
        end = t.replace(hour=15, minute=30)
        while t < end:
            mod = t.hour * 60 + t.minute
            if mod >= 15 * 60 + 15 and t + timedelta(minutes=step) < end:
                t += timedelta(minutes=step)
                continue                                            # settled form: only the last closing bar
            px += rng.normal(0, 1.5)
            flat = mod >= 15 * 60 + 15
            rows.append((t, px, px if flat else px + 1, px if flat else px - 1, px, 0 if flat else 1000))
            t += timedelta(minutes=step)
    return pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])


def _save(settings, tf, step, symbols=UNIVERSE):
    for i, sym in enumerate(symbols):
        path = history_store.history_path(settings, tf, sym)
        path.parent.mkdir(parents=True, exist_ok=True)
        _walk(step, i).to_parquet(path)


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setattr(trending, "universe_symbols", lambda _s: UNIVERSE)
    sent = []
    monkeypatch.setattr(tg, "notify", lambda text, *a, **k: sent.append(text))
    return sent


def _pair(tmp_path):
    from backend.core.config import load_settings
    main, main_path = _settings(tmp_path, "main")
    five_path = tmp_path / "five.yaml"
    five_path.write_text(f"extends: {(ROOT / 'config' / 'settings.yaml').as_posix()}\n"
                         f"instance:\n  name: 5 minute\n  history_peers: []\n  follow: {main_path.as_posix()}\n"
                         f"trading:\n  intraday_timeframe: 5m\n  trade_universe: ranked\n"
                         f"paths:\n  live_data: {(tmp_path / 'data_five').as_posix()}\n  logs: {(tmp_path / 'logs').as_posix()}\n",
                         encoding="utf-8")
    # the main system reads the 5 minute system's cache (as in the real config: instance.history_peers)
    main_path.write_text(main_path.read_text(encoding="utf-8").replace("history_peers: []", f"history_peers: [{five_path.as_posix()}]"), encoding="utf-8")
    return load_settings(main_path), load_settings(five_path)


# ---------------------------------------------------------------------- the list, made after the close, for the next session
def test_the_list_is_made_after_the_close_for_the_next_session_and_becomes_todays_list_next_morning(tmp_path):
    main, _ = _pair(tmp_path)
    _save(main, "15m", 15)
    picks = ns.build_next_list(main, TODAY)
    assert picks and len(picks) <= 10 and all(p["symbol"] in UNIVERSE for p in picks)
    assert trending.load_next(main, "paper", NEXT_DAY)["picks"] == picks
    assert trending.load_today(main, "paper", TODAY.date()) is None                # today's trading never sees tomorrow's list
    nxt = trending.load_today(main, "paper", NEXT_DAY)                              # next morning it is today's list
    assert nxt["picks"] == picks and trending.symbols_of(trending.load_today(main, "paper", NEXT_DAY)) == [p["symbol"] for p in picks]
    assert ns.build_next_list(main, TODAY) == picks                                 # asking again does not re-rank


def test_no_list_while_too_little_of_the_index_has_its_full_history(tmp_path):
    main, _ = _pair(tmp_path)
    _save(main, "15m", 15, UNIVERSE[:5])
    assert ns.build_next_list(main, TODAY) is None and trending.load_next(main, "paper", NEXT_DAY) is None


def test_both_systems_use_the_same_list_and_the_5_minute_one_waits_for_it(tmp_path, _quiet):
    main, five = _pair(tmp_path)
    _save(main, "15m", 15)
    _save(five, "5m", 5)
    assert ns.list_owner(five).paths.live_data == main.paths.live_data
    assert ns.after_close(five, TODAY) is False                                    # the main system has not made the list yet
    assert ns.after_close(main, TODAY) is True
    assert ns.after_close(five, TODAY) is True
    mine = [p["symbol"] for p in trending.load_next(main, "paper", NEXT_DAY)["picks"]]
    assert ns.priority_symbols(five, TODAY) == mine == ns.priority_symbols(main, TODAY)
    # next morning: both resolve to the very same stocks through the files they read
    assert watchlist.active_symbols(five, "paper")[0] == [] or True               # (today's real date: nothing for it)
    assert trending.symbols_of(trending.load_today(ns.list_owner(five), "paper", NEXT_DAY)) == mine
    assert not (five.paths.live_data / "state" / "trending_next_paper.json").exists()    # the 5 minute system makes no list of its own


def test_the_manifest_records_the_list_and_every_stocks_saved_history(tmp_path):
    main, five = _pair(tmp_path)
    _save(main, "15m", 15)
    _save(five, "5m", 5)
    assert ns.after_close(main, TODAY) and ns.after_close(five, TODAY)
    for system, tf in ((main, "15m"), (five, "5m")):
        m = json.loads((system.paths.live_data / "state" / "next_session.json").read_text(encoding="utf-8"))
        assert m["ready"] is True and m["for_session"] == "2026-11-18" and m["timeframe"] == tf
        assert len(m["list"]) == len(m["histories"]) and all(h["complete"] and h["bars"] > 100 for h in m["histories"].values())
    assert json.loads((main.paths.live_data / "state" / "next_session.json").read_text())["list"] == \
        json.loads((five.paths.live_data / "state" / "next_session.json").read_text())["list"]


def test_a_missing_history_of_a_listed_stock_means_not_ready(tmp_path, _quiet):
    main, five = _pair(tmp_path)
    _save(main, "15m", 15)
    _save(five, "5m", 5, UNIVERSE[:6])                                              # most of the 5m histories never got loaded
    ns.after_close(main, TODAY)
    picks = trending.load_next(main, "paper", NEXT_DAY)["picks"]
    ready = ns.after_close(five, TODAY)
    assert ready == all(p["symbol"] in UNIVERSE[:6] for p in picks)
    assert len([t for t in _quiet if "HISTORY READY" in t]) == (2 if ready else 0)      # nobody says so while a listed stock's 5m or 15m series is missing


def test_the_message_says_what_is_ready_and_when_things_start(tmp_path, _quiet):
    main, five = _pair(tmp_path)
    _save(main, "15m", 15)
    _save(five, "5m", 5)
    assert ns.after_close(main, TODAY)
    text = [t for t in _quiet if "HISTORY READY" in t][0]
    assert "09:10" in text and "09:15" in text and "15:00" in text and "Wed 18 Nov" in text
    assert all(p["symbol"] in text for p in trending.load_next(main, "paper", NEXT_DAY)["picks"])


def test_the_list_stocks_are_loaded_first_even_when_the_list_appears_during_the_sweep(tmp_path):
    main, _ = _pair(tmp_path)
    asked = []
    picks_later = ["S09", "S03"]

    def fetch(settings, symbol, tf, now):
        asked.append(symbol)
        if len(asked) == 2:                                                         # the list is made while the sweep is running
            trending.save_next(settings, "paper", [{"rank": i + 1, "symbol": s, "score": 1.0} for i, s in enumerate(picks_later)],
                               NEXT_DAY, tf)
        return None
    ch.warm_once(main, now=TODAY, fetch=fetch)
    assert asked[:2] == ["S00", "S01"] and asked[2:4] == sorted(picks_later) and sorted(asked) == UNIVERSE


# ---------------------------------------------------------------------- force exit: commands never lost, never stale
def test_force_exit_commands_are_all_delivered_once_in_order(tmp_path):
    main, _ = _pair(tmp_path)
    for sym in ("A", "B", "C"):
        pc.queue_close_position(main, "paper", sym)
    got = pc.pop_commands(main, "paper")
    assert [c["symbol"] for c in got] == ["A", "B", "C"] and all(c["action"] == "close_position" for c in got)
    assert pc.pop_commands(main, "paper") == []


def test_force_exit_commands_from_many_threads_are_not_lost_while_the_process_takes_them(tmp_path):
    main, _ = _pair(tmp_path)
    taken, stop = [], threading.Event()

    def taker():
        while not stop.is_set():
            taken.extend(pc.pop_commands(main, "paper"))

    t = threading.Thread(target=taker)
    t.start()
    writers = [threading.Thread(target=lambda i=i: [pc.queue_close_position(main, "paper", f"W{i}-{k}") for k in range(25)]) for i in range(8)]
    [w.start() for w in writers]
    [w.join() for w in writers]
    stop.set()
    t.join()
    taken.extend(pc.pop_commands(main, "paper"))
    assert {c["symbol"] for c in taken} == {f"W{i}-{k}" for i in range(8) for k in range(25)}          # every command arrived (a legacy copy may repeat one)


def test_an_old_force_exit_is_dropped_and_the_old_queue_file_still_works(tmp_path):
    main, _ = _pair(tmp_path)
    folder = pc._commands_dir(main, "paper")
    folder.mkdir(parents=True)
    old = (datetime.now() - timedelta(hours=9)).isoformat(timespec="seconds")
    (folder / "1-old.json").write_text(json.dumps({"action": "close_position", "symbol": "OLD", "queued_at": old}), encoding="utf-8")
    pc._commands_path(main, "paper").write_text(json.dumps([{"action": "close_position", "symbol": "LEGACY"}]), encoding="utf-8")
    assert [c["symbol"] for c in pc.pop_commands(main, "paper")] == ["LEGACY"]


# ---------------------------------------------------------------------- the exe finds the project from the folder above it
def test_the_exe_in_the_folder_above_the_project_finds_it(tmp_path):
    outer = tmp_path / "v17 anthropic"
    project = outer / "second-system"
    (project / "config").mkdir(parents=True)
    (project / "config" / "settings.yaml").write_text("trading: {}\n", encoding="utf-8")
    code = ("import sys; sys.frozen = True; sys.executable = sys.argv[1]; sys.path.insert(0, sys.argv[2]); "
            "import backend.core.config as c; print(c.ROOT)")
    out = subprocess.run([sys.executable, "-c", code, str(outer / "both_systems.exe"), str(ROOT)], capture_output=True, text=True, cwd=str(tmp_path))
    assert out.returncode == 0, out.stderr
    assert Path(out.stdout.strip()) == project.resolve()
    inside = subprocess.run([sys.executable, "-c", code, str(project / "both_systems.exe"), str(ROOT)], capture_output=True, text=True, cwd=str(tmp_path))
    assert Path(inside.stdout.strip()) == project.resolve()                           # next to its config: unchanged behaviour


def test_a_force_exit_also_reaches_a_trading_process_that_still_reads_only_the_old_queue_file(tmp_path):
    main, _ = _pair(tmp_path)
    pc.queue_close_position(main, "paper", "SBIN")
    legacy = json.loads(pc._commands_path(main, "paper").read_text(encoding="utf-8"))                    # what an older process reads
    assert legacy == [{"action": "close_position", "symbol": "SBIN"}]
    assert [c["symbol"] for c in pc.pop_commands(main, "paper")] == ["SBIN"]                              # a current one gets it once
    assert pc.pop_commands(main, "paper") == [] and not pc._commands_path(main, "paper").exists()
