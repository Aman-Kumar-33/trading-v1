"""Research recorder: every closed bar of a listed stock is saved with its indicators, the MACD cross analysis and
the system's decision, ticks are saved, bars that closed before a stock was listed are backfilled, and nothing
is written for stocks that are not on the list.

Run from second-system/:  python -m pytest tests -q
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_macd_system import (PER_DAY, build_system, make_closes, run_candles, session_starts)  # noqa: E402

from backend.core.clock import IST  # noqa: E402
from backend.core.models import Tick  # noqa: E402
from backend.data.macd_recorder import BAR_FIELDS, MacdRecorder, analyse  # noqa: E402
from backend.features.pipeline import build_features  # noqa: E402
from backend.strategies.intraday.macd_only import macd_cross_masks  # noqa: E402

N, WARM = 300, 100


def _run(tmp_path, listed=True):
    closes = make_closes(N)
    starts = session_starts(N // PER_DAY + 1)[:N]
    bus, paper, om, runner, log = build_system(closes, starts, WARM)
    runner.recorder = MacdRecorder(tmp_path, "15m", starts[-1].date())
    runner.set_watchlist({"TEST"} if listed else {"OTHER"})
    run_candles(bus, paper, closes, starts, WARM, N)
    runner.recorder.close()
    return runner, starts, closes, om


def test_every_closed_bar_is_saved_with_macd_analysis_and_decisions(tmp_path):
    runner, starts, closes, om = _run(tmp_path)
    df = pd.read_csv(tmp_path / starts[-1].date().isoformat() / "bars_15m" / "TEST.csv")
    assert list(df.columns) == BAR_FIELDS and len(df) == N - WARM
    assert list(df["bar_start"]) == [s.isoformat() for s in starts[WARM:]]
    assert (df["source"] == "live").all()
    # the analysis columns agree with the numbers the strategy used
    assert ((df["macd"] - df["macd_signal"]) - df["gap"]).abs().max() < 1e-9
    assert (df["gap"].shift(1).iloc[1:] - df["prev_gap"].iloc[1:]).abs().max() < 1e-9
    bull, bear = macd_cross_masks(df["macd"].to_numpy(), df["macd_signal"].to_numpy())
    assert list(df["cross"].fillna("")[bull]) == ["BULL"] * int(bull.sum()) and int(bull.sum()) > 0
    assert list(df["cross"].fillna("")[bear]) == ["BEAR"] * int(bear.sum())
    # every cross bar has the system's decision next to it, and the bought bars show the trade was taken
    crosses = df[df["cross"].notna()]
    assert (crosses["signal"] == crosses["cross"].map({"BULL": "BUY", "BEAR": "SELL"})).all()
    assert any(str(a).startswith("BOUGHT") for a in crosses["action"])
    sig_file = pd.read_csv(tmp_path / starts[-1].date().isoformat() / "signals_and_actions.csv")
    assert len(sig_file) >= len(crosses) and {"BUY", "SELL"} <= set(sig_file["side"])
    assert df["ema_12"].notna().all() and df["gap_in_atr"].notna().all()


def test_unlisted_stock_is_not_recorded(tmp_path):
    _run(tmp_path, listed=False)
    day = next(tmp_path.iterdir())
    assert not (day / "bars_15m").exists() and not (day / "ticks").exists()


def test_ticks_are_saved_only_for_listed_stocks(tmp_path):
    runner, starts, closes, om = _run(tmp_path)
    import asyncio
    from backend.core import events

    async def go():
        for sym, px in (("TEST", 1001.5), ("OTHER", 50.0)):
            await runner.bus.publish(events.TICK, Tick(symbol=sym, ts=starts[-1].replace(hour=10), ltp=px, cum_volume=123, last_qty=5))
    asyncio.run(go())
    runner.recorder.close()
    day = tmp_path / starts[-1].date().isoformat() / "ticks"
    t = pd.read_csv(day / "TEST.csv")
    assert list(t.columns) == ["ts", "ltp", "cum_volume", "last_qty"] and t["ltp"].iloc[0] == 1001.5
    assert not (day / "OTHER.csv").exists()


def test_backfill_writes_todays_earlier_bars_once(tmp_path):
    closes = make_closes(N)
    starts = session_starts(N // PER_DAY + 1)[:N]
    bus, paper, om, runner, log = build_system(closes, starts, WARM)
    runner.set_watchlist(set())                                    # auto mode, not picked yet: bars kept, nothing saved
    today = starts[WARM + 5].date()
    runner.recorder = MacdRecorder(tmp_path, "15m", today)
    run_candles(bus, paper, closes, starts, WARM, WARM + 12)
    assert not (tmp_path / today.isoformat() / "bars_15m").exists()
    runner.set_watchlist({"TEST"})                                 # picked
    runner.backfill_recording(["TEST"])
    runner.backfill_recording(["TEST"])                            # idempotent
    df = pd.read_csv(tmp_path / today.isoformat() / "bars_15m" / "TEST.csv")
    want = [s.isoformat() for s in starts[:WARM + 12] if s.date() == today]
    assert list(df["bar_start"]) == want and (df["source"] == "backfill").all()


def test_official_bars_are_compared_with_live(tmp_path):
    runner, starts, closes, om = _run(tmp_path)
    rec = MacdRecorder(tmp_path, "15m", starts[-1].date())
    official = pd.DataFrame({"ts": starts, "open": closes, "high": closes + 1, "low": closes - 1, "close": closes,
                             "volume": 10_000.0})
    row = rec.save_official("TEST", build_features(official, "15m", core_only=True))
    rec.save_comparison([row])
    assert row["bars_compared"] > 0 and row["max_close_diff"] == 0 and row["crosses_only_live"] == row["crosses_only_official"] == ""
    off = pd.read_csv(tmp_path / starts[-1].date().isoformat() / "official_15m" / "TEST.csv")
    assert {"gap", "cross", "ema_12"} <= set(off.columns)
    assert (tmp_path / starts[-1].date().isoformat() / "live_vs_official.csv").exists()


def test_analyse_flags_the_same_crosses_as_the_strategy():
    closes = make_closes(200)
    starts = session_starts(200 // PER_DAY + 1)[:200]
    feat = build_features(pd.DataFrame({"ts": starts, "open": closes, "high": closes + 1, "low": closes - 1,
                                        "close": closes, "volume": 10_000.0}), "15m", core_only=True)
    ana = analyse(feat)
    bull, bear = macd_cross_masks(feat["macd"].to_numpy(), feat["macd_signal"].to_numpy())
    assert list(ana["cross"][bull]) == ["BULL"] * int(bull.sum()) and list(ana["cross"][bear]) == ["BEAR"] * int(bear.sum())
    assert (ana["cross"][~(bull | bear)] == "").all()
