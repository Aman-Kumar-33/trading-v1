"""Paper trading v2: system-decided 5-25% sizing, max 10 open positions, the daily trending picker, auto mode
(empty stock box), and the Telegram start / picks messages.

Run from second-system/:  python -m pytest tests -q
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_macd_system import (PER_DAY, build_system, frame, make_closes, run_candles, session_starts)  # noqa: E402

from backend.core.clock import IST  # noqa: E402
from backend.core.config import Settings  # noqa: E402
from backend.core.models import Side  # noqa: E402
from backend.data import trending, watchlist  # noqa: E402
from backend.execution import sizing  # noqa: E402
from backend.execution.order_manager import Position  # noqa: E402
from backend.notify import telegram as tg  # noqa: E402
from backend.strategies.intraday.macd_only import MacdOnly  # noqa: E402

CAP = 1_000_000.0


@pytest.fixture
def sent(monkeypatch):
    messages: list[str] = []
    monkeypatch.setattr(tg, "notify", messages.append)
    return messages


def make_settings(tmp_path, trading=None) -> Settings:
    return Settings({"paths": {"backtest_data": str(tmp_path / "bt"), "live_data": str(tmp_path / "data"),
                               "logs": str(tmp_path / "logs")}, "trading": trading or {}}, root=tmp_path)


# ---------------------------------------------------------------------- sizing (5-25% decided per signal)
def test_size_tiers_follow_conviction_and_never_exceed_the_ceiling():
    strong = sizing.SizeContext(hist_jump=0.2, atr=1.0, trend_score=95)      # steep cross, top pick, calm (atr 0.1% of price)
    weak = sizing.SizeContext(hist_jump=0.001, atr=8.0, trend_score=5)       # barely crossed, bottom pick, jumpy
    mid = sizing.SizeContext(hist_jump=0.03, atr=1.0, trend_score=50)
    assert sizing.choose_pct(strong, 1000.0, 0.10, 0.05, 0.25)[0] == 0.25
    assert sizing.choose_pct(weak, 1000.0, 0.10, 0.05, 0.25)[0] == 0.05
    assert sizing.choose_pct(mid, 1000.0, 0.10, 0.05, 0.25)[0] in (0.10, 0.15)
    assert sizing.choose_pct(None, 1000.0, 0.10, 0.05, 0.25)[0] == 0.10      # no data -> default size
    assert sizing.choose_pct(strong, 1000.0, 0.10, 0.05, 0.15)[0] == 0.15    # a lower configured ceiling wins


def test_plan_size_caps_at_25_percent_and_buys_whole_shares():
    p = sizing.plan_size(1000.0, CAP, CAP, 0, 10, 0.25, 0.05, 0.25)
    assert p.qty == 250 and p.pct_of_capital <= 0.25
    p = sizing.plan_size(1000.0, CAP, CAP, 0, 10, 0.90, 0.05, 0.25)
    assert p.qty == 250                                                       # a bigger ask is cut to the hard ceiling


def test_plan_size_is_limited_by_free_cash_and_skips_when_too_little():
    # 2 slots left, free cash 300k -> at most 150k (a share of the cash is kept for the slots still open)
    p = sizing.plan_size(1000.0, CAP, 300_000.0, 8, 10, 0.25, 0.05, 0.25)
    assert p.qty == 150
    p = sizing.plan_size(1000.0, CAP, 30_000.0, 9, 10, 0.25, 0.05, 0.25)
    assert p.qty == 0 and "not enough free cash" in p.skip_reason            # under the 5% minimum trade
    p = sizing.plan_size(80_000.0, CAP, CAP, 0, 10, 0.05, 0.05, 0.25)        # one share costs more than the 50k budget
    assert p.qty == 0 and "one share" in p.skip_reason                        # (the old max(1, ..) bought it anyway)


def _fill_other_positions(om, count):
    for k in range(count):
        om.open_positions.append(Position(f"HELD{k}", Side.BUY, MacdOnly.horizon, 10, 100.0, 1.0, None, "macd_only", "v1",
                                          None, datetime(2026, 10, 1, 10, 0, tzinfo=IST), product="CNC"))


def test_buy_is_skipped_when_ten_positions_are_open(sent):
    n = 300
    closes = make_closes(n)
    starts = session_starts(n // PER_DAY + 1)[:n]
    bus, paper, om, runner, log = build_system(closes, starts, 100)
    runner.max_open_positions = 10                                            # a limit can still be configured (0 = none, the default)
    _fill_other_positions(om, 10)
    run_candles(bus, paper, closes, starts, 100, n)
    assert not om.has_position("TEST") and not om.closed_trades
    assert any(why == "max open positions 10/10" for _, _, _, why in log)
    assert any("SKIPPED - 10/10 positions already open" in m for m in sent)


def test_buy_is_taken_when_a_slot_is_free_and_sized_inside_5_to_25_percent(sent):
    n = 300
    closes = make_closes(n)
    starts = session_starts(n // PER_DAY + 1)[:n]
    bus, paper, om, runner, log = build_system(closes, starts, 100)
    runner.max_open_positions = 10
    _fill_other_positions(om, 9)                                              # one slot left
    run_candles(bus, paper, closes, starts, 100, n)
    trades = [t.position for t in om.closed_trades] + [p for p in om.open_positions if p.symbol == "TEST"]
    assert trades, "the seed has bullish crosses, so a buy must happen"
    assert all(0.04 <= t.qty * t.entry_price / CAP <= 0.25 for t in trades)
    assert any("positions open)" in m and "of capital" in m for m in sent)


def test_unlisted_stock_never_buys(sent):
    n = 300
    closes = make_closes(n)
    starts = session_starts(n // PER_DAY + 1)[:n]
    bus, paper, om, runner, log = build_system(closes, starts, 100)
    runner.set_watchlist({"OTHER"})                                           # TEST is not allowed to be bought
    run_candles(bus, paper, closes, starts, 100, n)
    assert not om.open_positions and not om.closed_trades


def test_bars_of_unlisted_stocks_are_kept_so_a_later_pick_has_its_first_bar(sent):
    n = 140
    closes = make_closes(n)
    starts = session_starts(n // PER_DAY + 1)[:n]
    bus, paper, om, runner, log = build_system(closes, starts, 100)
    runner.set_watchlist(set())                                               # auto mode before the 09:31 pick
    run_candles(bus, paper, closes, starts, 100, 110)
    assert len(runner._today_bars[("TEST", "15m")]) == 10 and not log
    runner.set_watchlist({"TEST"})                                            # picked: the earlier bars are in the series
    assert len(runner._build_df(("TEST", "15m"))) == 110


# ---------------------------------------------------------------------- trending picker + auto mode
def _history(kind: str, days: int = 14, seed: int = 0, last_price: float = 500.0) -> pd.DataFrame:
    starts = session_starts(days)
    t = np.arange(len(starts))
    if kind == "wave":      # clean swings -> many MACD crosses, trend-following earns money
        closes = last_price + 20 * np.sin(t / 5.0)
    elif kind == "flat":
        closes = np.full(len(starts), last_price)
    else:
        closes = last_price + np.cumsum(np.random.default_rng(seed).normal(0, 0.3, len(starts)))
    return frame(starts, closes)


def test_ranking_prefers_stocks_with_crosses_and_profit_and_drops_dead_ones():
    hist = {"WAVE": _history("wave"), "FLAT": _history("flat"), "WALK": _history("walk", seed=2)}
    picks = trending.rank_picks(hist, {}, count=10)
    names = [p["symbol"] for p in picks]
    assert "FLAT" not in names                                                # no crosses -> nothing to trade
    assert names[0] == "WAVE" and [p["rank"] for p in picks] == list(range(1, len(picks) + 1))
    assert picks[0]["crosses_per_session"] > 0 and picks[0]["net_return_pct"] > 0
    assert len(trending.rank_picks(hist, {}, count=1)) == 1


def test_ranking_uses_todays_quote_and_the_price_cap():
    hist = {"WAVE": _history("wave"), "WAVE2": _history("wave", last_price=900.0)}
    quotes = {"WAVE": {"ltp": 500.0, "high": 520.0, "low": 480.0, "percentChange": 4.0, "tradeVolume": 5_000_000},
              "WAVE2": {"ltp": 900.0, "high": 901.0, "low": 899.0, "percentChange": 0.1, "tradeVolume": 100}}
    picks = trending.rank_picks(hist, quotes, count=10)
    assert picks[0]["symbol"] == "WAVE" and picks[0]["pct_change"] == 4.0
    assert [p["symbol"] for p in trending.rank_picks(hist, quotes, count=10, max_price=600.0)] == ["WAVE"]


def test_trending_file_is_only_used_on_the_day_it_was_picked(tmp_path):
    settings = make_settings(tmp_path)
    picks = [{"rank": 1, "symbol": "SBIN", "score": 80.0}, {"rank": 2, "symbol": "TCS", "score": 70.0}]
    trending.save_picks(settings, "paper", picks, date(2026, 11, 16), "15m")
    assert trending.symbols_of(trending.load_today(settings, "paper", date(2026, 11, 16))) == ["SBIN", "TCS"]
    assert trending.load_today(settings, "paper", date(2026, 11, 17)) is None  # next day: re-pick, never reuse


def test_empty_stock_box_means_auto_and_a_given_list_wins(tmp_path):
    settings = make_settings(tmp_path, {"symbols": []})
    today = datetime.now().date()
    assert watchlist.is_auto_mode(settings, "paper")                          # nothing saved, nothing configured
    assert watchlist.active_symbols(settings, "paper")[0] == []               # auto, but not picked yet
    trending.save_picks(settings, "paper", [{"rank": 1, "symbol": "SBIN", "score": 80.0}], today, "15m")
    assert watchlist.active_symbols(settings, "paper")[0] == ["SBIN"]
    watchlist.save(settings, "paper", [])                                     # an empty saved box is still auto
    assert watchlist.is_auto_mode(settings, "paper")
    watchlist.save(settings, "paper", ["tcs"])
    assert not watchlist.is_auto_mode(settings, "paper") and watchlist.active_symbols(settings, "paper")[0] == ["TCS"]
    assert watchlist.is_auto_mode(settings, "live")                           # modes are separate


def test_startup_and_picks_messages():
    msg = tg.format_startup("TRADING SYSTEM", "PAPER", "15m", "macd_only", 1_000_000.0, 0.05, 0.25, 10,
                            ["SBIN", "TCS"], "auto - trending", "2026-11-17 09:01:00")
    for needle in ("STARTED", "PAPER", "15m", "macd_only", "1,000,000.00", "5% to 25%", "max 10", "SBIN, TCS", "auto - trending"):
        assert needle in msg
    assert "none yet" in tg.format_startup("TRADING SYSTEM", "PAPER", "15m", "macd_only", 1e6, 0.05, 0.25, 10, [], "auto", "t")
    picks = tg.format_picks("PAPER", "15m", [{"rank": 1, "symbol": "SBIN", "score": 80.0, "crosses_per_session": 2.1,
                                              "net_return_pct": 3.2, "pct_change": 1.4}], "2026-11-17 09:31:00")
    assert "1. SBIN" in picks and "+3.2%" in picks and "Only these will be bought" in picks


# ---------------------------------------------------------------------- a cross on the bar that closed just before the pick
def _cross_system(listed_at_close: bool):
    from test_macd_system import reference_cross_bars
    n = 300
    closes = make_closes(n)
    starts = session_starts(n // PER_DAY + 1)[:n]
    bull = [i for i in reference_cross_bars(closes)[0] if i >= 140 and starts[i].time() < datetime(2026, 1, 1, 14, 30).time()][0]
    bus, paper, om, runner, log = build_system(closes, starts, 100)
    runner.set_watchlist({"TEST"} if listed_at_close else set())
    run_candles(bus, paper, closes, starts, 100, bull + 1)          # the cross bar is the newest closed bar
    return runner, om, starts, bull, closes


def test_cross_just_before_the_pick_is_bought_at_the_pick(sent):
    import asyncio
    runner, om, starts, bull, closes = _cross_system(listed_at_close=False)
    assert not om.open_positions
    runner.set_watchlist({"TEST"})                                    # picked a minute after the bar closed
    now = starts[bull] + timedelta(minutes=16)
    asyncio.run(runner.catch_up_entries(["TEST"], now))
    assert om.has_position("TEST")
    assert any("caught up" in m or "BOUGHT" in m for m in sent)
    asyncio.run(runner.catch_up_entries(["TEST"], now))   # asking again never buys twice
    assert len(om.open_positions) == 1


def test_catch_up_ignores_old_bars_and_bars_already_evaluated(sent):
    import asyncio
    runner, om, starts, bull, closes = _cross_system(listed_at_close=False)
    runner.set_watchlist({"TEST"})
    late = starts[bull] + timedelta(minutes=15 + 20)   # more than a bar after the close: stale
    asyncio.run(runner.catch_up_entries(["TEST"], late))
    assert not om.open_positions
    runner2, om2, starts2, bull2, _ = _cross_system(listed_at_close=True)  # listed all along: the cross was already handled
    n_before = len(om2.open_positions) + len(om2.closed_trades)
    asyncio.run(runner2.catch_up_entries(["TEST"], starts2[bull2] + timedelta(minutes=16)))
    assert len(om2.open_positions) + len(om2.closed_trades) == n_before and n_before >= 1
