"""MACD-only system: signal timing, long/flat behaviour, closing-bar handling, MACD-only exits, Telegram text.

Run from second-system/:  python -m pytest tests -q
"""
from __future__ import annotations

import asyncio
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.brokers.paper import PaperBroker
from backend.core import events
from backend.core.clock import IST
from backend.core.events import EventBus
from backend.core.models import Candle, Mode, Side, Tick
from backend.data.history_store import drop_forming
from backend.execution.order_manager import OrderManager, Position
from backend.execution.strategy_runner import SimpleRunner
from backend.features import indicators as ind
from backend.features.pipeline import build_features
from backend.notify import telegram as tg
from backend.risk.risk_manager import RiskLimits, RiskManager
from backend.strategies import ALL_STRATEGIES
from backend.strategies.base import StrategyContext
from backend.strategies.intraday.macd_only import MacdOnly, macd_cross_masks

TF = "15m"
STEP = 15                  # minutes per bar
PER_DAY = 24               # 09:15 ... 15:00 bar starts (+ the 15:15 bar when closing_bar=True)


# ---------------------------------------------------------------------- helpers
def session_starts(days: int, first: date = date(2026, 10, 5), closing_bar: bool = False) -> list[datetime]:
    """15m bar start times 09:15 ... 15:00 (+ the 15:15-15:30 closing bar) over `days` weekdays."""
    per_day = PER_DAY + (1 if closing_bar else 0)
    out, d = [], first
    while len(out) < days * per_day:
        if d.weekday() < 5:
            t = datetime(d.year, d.month, d.day, 9, 15, tzinfo=IST)
            for _ in range(per_day):
                out.append(t)
                t += timedelta(minutes=STEP)
        d += timedelta(days=1)
    return out


def make_closes(n: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 1000 + np.cumsum(rng.normal(0, 3, n))


def candle(symbol: str, start: datetime, close: float) -> Candle:
    end = min(start + timedelta(minutes=STEP), start.replace(hour=15, minute=30))
    return Candle(symbol, TF, start, end, close, close + 1, close - 1, close, volume=10_000, closed=True)


def frame(starts: list[datetime], closes: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"ts": starts, "open": closes, "high": closes + 1, "low": closes - 1, "close": closes,
                         "volume": 10_000.0})


def build_system(closes: np.ndarray, starts: list[datetime], warm: int, deferred_file=None):
    bus = EventBus()
    paper = PaperBroker(bus, starting_cash=1_000_000)
    om = OrderManager(paper, RiskManager(RiskLimits(capital=1_000_000)), mode=Mode.PAPER,
                      profit_lock_stop=False, trail_stop=False)
    signals_log: list[tuple] = []
    runner = SimpleRunner(bus, {TF: [MacdOnly(version="v1")]}, {}, om, capital=1_000_000, history_bars=500,
                          signal_sink=lambda sig, status, why: signals_log.append((sig.side.value, str(sig.ts), status, why)),
                          deferred_file=deferred_file)
    runner.set_history("TEST", TF, frame(starts[:warm], closes[:warm]))
    return bus, paper, om, runner, signals_log


def reference_cross_bars(closes: np.ndarray) -> tuple[list[int], list[int]]:
    """Independent MACD (SMA-seeded EMAs, like TradingView) -> indices of bullish / bearish cross bars."""
    def ema(v, n):
        out = np.full(len(v), np.nan)
        first = int(np.argmax(~np.isnan(v)))
        k = 2 / (n + 1)
        out[first + n - 1] = v[first:first + n].mean()
        for i in range(first + n, len(v)):
            out[i] = v[i] * k + out[i - 1] * (1 - k)
        return out
    macd = ema(closes, 12) - ema(closes, 26)
    sig = ema(macd, 9)
    bull, bear = macd_cross_masks(macd, sig)
    return list(np.flatnonzero(bull)), list(np.flatnonzero(bear))


# ---------------------------------------------------------------------- cross detection
def test_cross_masks_mark_exactly_the_completing_bar():
    macd = np.array([np.nan, -1.0, -0.5, 0.2, 0.6, 0.1, -0.3, -0.4])
    sig = np.array([np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    bull, bear = macd_cross_masks(macd, sig)
    assert list(np.flatnonzero(bull)) == [3]     # -0.5 (below) -> 0.2 (above)
    assert list(np.flatnonzero(bear)) == [6]     # 0.1 (above) -> -0.3 (below)


def test_cross_masks_ignore_warmup_nan_and_wraparound():
    macd = np.array([5.0, np.nan, 1.0, 2.0])
    sig = np.array([0.0, np.nan, 3.0, 1.0])
    bull, bear = macd_cross_masks(macd, sig)
    assert not bull[0] and not bear[0]           # bar 0 has no predecessor (np.roll wraps the last bar to 0)
    assert not bull[2] and not bear[2]           # predecessor is NaN
    assert list(np.flatnonzero(bull)) == [3] or not bull.any()


def test_strategy_signals_match_independent_macd_on_every_cross():
    n = 400
    closes = make_closes(n)
    starts = session_starts(n // PER_DAY + 1)[:n]
    feat = build_features(frame(starts, closes), TF, core_only=True)
    sigs = MacdOnly(version="v1").run(StrategyContext("TEST", TF, feat))
    ref_bull, ref_bear = reference_cross_bars(closes)
    skip = 120   # let the pandas-seeded and SMA-seeded EMAs converge before comparing
    got_bull = [i for i, t in enumerate(starts) if any(s.ts == t and s.side == Side.BUY for s in sigs) and i >= skip]
    got_bear = [i for i, t in enumerate(starts) if any(s.ts == t and s.side == Side.SELL for s in sigs) and i >= skip]
    assert got_bull == [i for i in ref_bull if i >= skip]
    assert got_bear == [i for i in ref_bear if i >= skip]
    assert len(got_bull) > 3 and len(got_bear) > 3   # the seed really produces crosses to compare


def test_every_signal_carries_macd_values_and_bar_time():
    closes = make_closes(200)
    starts = session_starts(20)[:200]
    feat = build_features(frame(starts, closes), TF, core_only=True)
    sigs = MacdOnly(version="v1").run(StrategyContext("TEST", TF, feat))
    assert sigs and all("MACD" in s.reason and s.ts in set(starts) for s in sigs)


# ---------------------------------------------------------------------- live runner
def run_candles(bus, paper, closes, starts, begin, end):
    async def go():
        for i in range(begin, end):
            paper.set_price("TEST", float(closes[i]))
            await bus.publish(events.CANDLE_CLOSED, candle("TEST", starts[i], float(closes[i])))
    asyncio.run(go())


def test_runner_trades_each_cross_long_flat_never_short():
    n, warm = 360, 100
    closes = make_closes(n)
    starts = session_starts(n // PER_DAY + 1)[:n]
    bus, paper, om, runner, _ = build_system(closes, starts, warm)
    run_candles(bus, paper, closes, starts, warm, n)
    ref_bull, ref_bear = reference_cross_bars(closes)
    skip = 130
    exp_bull = [i for i in ref_bull if i >= skip]
    got_entries = sorted(t.position.opened_at for t in om.closed_trades) + [p.opened_at for p in om.open_positions]
    # entry time = close of the cross bar = start + one bar
    got_entry_bars = [starts.index(t - timedelta(minutes=STEP)) for t in got_entries]
    assert [b for b in got_entry_bars if b >= skip] == exp_bull
    assert all(t.position.side == Side.BUY for t in om.closed_trades) and all(p.side == Side.BUY for p in om.open_positions)
    assert len(om.open_positions) <= 1


def test_sell_while_flat_is_ignored_not_shorted():
    # Crosses alternate, so a SELL with no long only happens when the run starts in an uptrend (restart,
    # or a BUY the system never saw). Start the live run two bars before a bearish cross to recreate that.
    n = 300
    closes = make_closes(n)
    starts = session_starts(n // PER_DAY + 1)[:n]
    _, ref_bear = reference_cross_bars(closes)
    bear = next(i for i in ref_bear if i >= 130)
    warm = bear - 2
    bus, paper, om, runner, log = build_system(closes, starts, warm)
    run_candles(bus, paper, closes, starts, warm, bear + 1)
    assert [(s, st, why) for s, _, st, why in log if s == "SELL"] == \
        [("SELL", "REJECTED", "SELL ignored - no open long (strategy is long/flat, never shorts)")]
    assert not om.open_positions and not om.closed_trades


def test_long_exits_on_bearish_state_even_if_the_cross_was_missed():
    # a long carried in from before a restart, MACD already below its signal and NOT crossing on this bar:
    # the strategy fires no SELL here, yet the long must still be flushed (it is the missed cross's exit)
    n = 220
    closes = make_closes(n, seed=11)
    starts = session_starts(n // PER_DAY + 1)[:n]
    full = build_features(frame(starts, closes), TF, core_only=True)
    _, ref_bear = reference_cross_bars(closes)
    bar = next(i for i in range(130, n) if full["macd"].iloc[i] < full["macd_signal"].iloc[i]
               and i not in ref_bear and i - 1 not in ref_bear)
    bus, paper, om, runner, _ = build_system(closes, starts, bar)
    om.open_positions.append(Position("TEST", Side.BUY, MacdOnly.horizon, 10, float(closes[bar - 1]), 1.0, None,
                                      "macd_only", "v1", None, starts[bar - 1], product="CNC", carried=True))
    run_candles(bus, paper, closes, starts, bar, bar + 1)
    assert not om.open_positions and [t.exit_reason for t in om.closed_trades] == ["SIGNAL"]


def test_long_is_kept_while_macd_stays_above_signal():
    n = 220
    closes = make_closes(n, seed=11)
    starts = session_starts(n // PER_DAY + 1)[:n]
    full = build_features(frame(starts, closes), TF, core_only=True)
    _, ref_bear = reference_cross_bars(closes)
    bar = next(i for i in range(130, n) if full["macd"].iloc[i] > full["macd_signal"].iloc[i]
               and full["macd"].iloc[i - 1] > full["macd_signal"].iloc[i - 1])
    bus, paper, om, runner, _ = build_system(closes, starts, bar)
    om.open_positions.append(Position("TEST", Side.BUY, MacdOnly.horizon, 10, float(closes[bar - 1]), 1.0, None,
                                      "macd_only", "v1", None, starts[bar - 1], product="CNC", carried=True))
    run_candles(bus, paper, closes, starts, bar, bar + 1)
    assert len(om.open_positions) == 1 and not om.closed_trades


def test_no_profit_lock_or_trailing_exits_when_disabled():
    closes = make_closes(120)
    starts = session_starts(12)[:120]
    bus, paper, om, runner, _ = build_system(closes, starts, 100)
    pos = Position("TEST", Side.BUY, MacdOnly.horizon, 10, 1000.0, 800.0, None, "macd_only", "v1", None, starts[99],
                   product="CNC", initial_risk=200.0)
    om.open_positions.append(pos)

    async def go():
        for px in (1010, 1030, 1400, 1005, 999):     # +2R, then back through breakeven - old code stopped out
            await om.on_tick(Tick("TEST", starts[100], float(px)))
    paper.set_price("TEST", 1000.0)
    asyncio.run(go())
    assert om.open_positions and pos.stop_loss == 800.0 and not om.closed_trades


def test_symbol_without_warmup_history_is_not_evaluated():
    bus = EventBus()
    paper = PaperBroker(bus, starting_cash=1_000_000)
    om = OrderManager(paper, RiskManager(RiskLimits(capital=1_000_000)), mode=Mode.PAPER)
    runner = SimpleRunner(bus, {TF: [MacdOnly(version="v1")]}, {}, om)
    starts = session_starts(2)
    paper.set_price("TEST", 1000.0)
    asyncio.run(bus.publish(events.CANDLE_CLOSED, candle("TEST", starts[0], 1000.0)))
    assert not om.open_positions and not runner.is_ready("TEST", TF)


def test_closing_bar_signal_executes_on_next_sessions_first_tick(tmp_path):
    n, warm = 260, 120
    closes = make_closes(n, seed=1)
    starts = session_starts(n // (PER_DAY + 1) + 1, closing_bar=True)[:n]
    ref_bull, _ = reference_cross_bars(closes)
    stub_cross = [i for i in ref_bull if i >= warm and starts[i].hour == 15 and starts[i].minute == 15]
    if not stub_cross:
        pytest.skip("seed has no bullish cross on a 15:15 closing bar")
    i = stub_cross[0]
    state = tmp_path / "deferred.json"
    bus, paper, om, runner, log = build_system(closes, starts, warm, deferred_file=state)
    run_candles(bus, paper, closes, starts, warm, i + 1)
    assert not om.open_positions, "nothing can be traded after the 15:30 close"
    assert any("closing 15:15 bar" in (why or "") for _, _, _, why in log)
    nxt = datetime(starts[i].year, starts[i].month, starts[i].day, 9, 15, tzinfo=IST) + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    # the process exits at 15:35 - a brand-new process must pick the signal up from disk
    bus2, paper2, om2, runner2, _ = build_system(closes, starts, warm, deferred_file=state)
    paper2.set_price("TEST", 1234.0)
    asyncio.run(bus2.publish(events.TICK, Tick("TEST", nxt, 1234.0)))
    assert len(om2.open_positions) == 1 and om2.open_positions[0].entry_price == 1234.0
    assert state.read_text() == "[]"   # consumed - a later restart must not re-trade it


# ---------------------------------------------------------------------- history / pool / telegram
def test_forming_bar_is_dropped_but_closing_bar_is_kept():
    day = datetime(2026, 11, 13, tzinfo=IST)
    starts = [day.replace(hour=14, minute=45), day.replace(hour=15, minute=15),
              (day + timedelta(days=3)).replace(hour=9, minute=15), (day + timedelta(days=3)).replace(hour=9, minute=30)]
    df = pd.DataFrame({"ts": starts, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1})
    now = (day + timedelta(days=3)).replace(hour=9, minute=40)        # 09:30 bar ends 09:45 -> still forming
    kept = drop_forming(df, TF, now)
    assert list(kept["ts"]) == starts[:3]
    assert len(drop_forming(df, TF, day.replace(hour=15, minute=29))) == 1     # closing bar needs the 15:30 close


def test_only_macd_is_in_the_strategy_pool():
    assert [c.strategy_id for c in ALL_STRATEGIES] == ["macd_only"]


def test_telegram_messages_carry_every_requested_field():
    entry = tg.format_entry("RELIANCE", "macd_only", "BUY", 38, 1310.2, "PAPER", "2026-11-16 11:30:00")
    for needle in ("RELIANCE", "macd_only", "BUY", "38", "1,310.20", "49,787.60", "P&L"):
        assert needle in entry
    exit_ = tg.format_exit("RELIANCE", "macd_only", "BUY", 38, 1325.0, 1310.2, 562.4, 64.5, 497.9, "SIGNAL",
                           "PAPER", "2026-11-16 13:45:00")
    for needle in ("RELIANCE", "macd_only", "SELL", "38", "1,325.00", "50,350.00", "+497.90"):
        assert needle in exit_


# ---------------------------------------------------------------------- stock list + per-signal Telegram
from backend.core.config import Settings
from backend.data import watchlist


def test_watchlist_parse_normalises_and_dedupes():
    assert watchlist.parse("reliance, tcs  hdfcbank.NS\nINFY-EQ;tcs") == ["RELIANCE", "TCS", "HDFCBANK", "INFY"]
    assert watchlist.parse(None) == [] and watchlist.parse("  ,, ") == []


def test_watchlist_resolution_order_and_no_universe_fallback(tmp_path):
    settings = Settings({"paths": {"backtest_data": str(tmp_path / "bt"), "live_data": str(tmp_path / "data"),
                                   "logs": str(tmp_path / "logs")}, "trading": {"symbols": ["SBIN"]}}, root=tmp_path)
    assert watchlist.resolve_symbols(settings, "paper")[0] == ["SBIN"]                    # settings.yaml
    watchlist.save(settings, "paper", ["tcs", "infy"])
    assert watchlist.resolve_symbols(settings, "paper")[0] == ["TCS", "INFY"]             # dashboard list wins
    assert watchlist.resolve_symbols(settings, "paper", "wipro")[0] == ["WIPRO"]          # --symbols wins
    watchlist.save(settings, "paper", [])
    assert watchlist.resolve_symbols(settings, "paper")[0] == []                          # saved-empty is respected
    assert watchlist.resolve_symbols(settings, "live")[0] == ["SBIN"]                     # modes are separate


@pytest.fixture
def sent(monkeypatch):
    messages: list[str] = []
    monkeypatch.setattr(tg, "notify", messages.append)
    return messages


def test_every_signal_sends_a_telegram_message_saying_what_happened(sent):
    n = 300
    closes = make_closes(n)
    starts = session_starts(n // PER_DAY + 1)[:n]
    _, ref_bear = reference_cross_bars(closes)
    bear = next(i for i in ref_bear if i >= 130)
    bus, paper, om, runner, log = build_system(closes, starts, bear - 2)      # starts flat just before a bearish cross
    run_candles(bus, paper, closes, starts, bear - 2, n)
    signal_msgs = [m for m in sent if "SIGNAL -" in m]
    assert len(signal_msgs) == len(log) and len(log) >= 3                      # one message per signal, acted on or not
    assert "ignored - no open long" in signal_msgs[0] and signal_msgs[0].startswith("SELL SIGNAL")
    assert any("BOUGHT" in m for m in signal_msgs) and any("SOLD" in m and "net P&L" in m for m in signal_msgs)
    for m in signal_msgs:
        for field in ("Stock: TEST", "Strategy: macd_only", "Signal: MACD", "Bar:", "Action:"):
            assert field in m
    assert any(m.startswith("BUY - PAPER") and "Total buy value" in m for m in sent)   # the fill alert still goes out


def test_stocks_not_on_the_list_are_ignored_entirely(sent):
    n = 300
    closes = make_closes(n)
    starts = session_starts(n // PER_DAY + 1)[:n]
    bus, paper, om, runner, log = build_system(closes, starts, 100)
    runner.set_watchlist({"SOMETHING_ELSE"})
    run_candles(bus, paper, closes, starts, 100, n)
    assert not log and not sent and not om.open_positions and not om.closed_trades


def test_unlisted_stock_with_an_open_long_still_exits_but_never_buys(sent):
    n = 300
    closes = make_closes(n)
    starts = session_starts(n // PER_DAY + 1)[:n]
    ref_bull, ref_bear = reference_cross_bars(closes)
    bar = next(i for i in ref_bear if i >= 130)
    bus, paper, om, runner, log = build_system(closes, starts, bar)
    runner.set_watchlist({"SOMETHING_ELSE"})                                   # TEST was removed from the list
    om.open_positions.append(Position("TEST", Side.BUY, MacdOnly.horizon, 10, float(closes[bar - 1]), 1.0, None,
                                      "macd_only", "v1", None, starts[bar - 1], product="CNC", carried=True))
    run_candles(bus, paper, closes, starts, bar, n)
    assert [t.exit_reason for t in om.closed_trades] == ["SIGNAL"]             # exit rule still applies
    assert not om.open_positions                                               # ...and no new BUY after it
    assert any("SELL SIGNAL" in m and "SOLD 10" in m for m in sent)


# ---------------------------------------------------------------------- history retry
from backend.data import history_store


def _run_load_all(symbols, fail_plan, **kw):
    """fail_plan: symbol -> number of times its fetch fails before succeeding (None = always fails)."""
    calls: list[str] = []
    loaded: list[str] = []
    attempts: dict[str, int] = {}

    async def fetch_one(sym):
        calls.append(sym)
        attempts[sym] = attempts.get(sym, 0) + 1
        limit = fail_plan.get(sym, 0)
        if limit is None or attempts[sym] <= limit:
            raise RuntimeError("Access denied because of exceeding access rate")
        return pd.DataFrame({"ts": [pd.Timestamp("2026-11-16 09:15", tz=IST)], "close": [1.0]})

    async def go():
        stop = asyncio.Event()
        left = await asyncio.wait_for(
            history_store.load_all(symbols, fetch_one, lambda s, df: loaded.append(s), stop, asyncio.Lock(),
                                   spacing=0, retry_delays=(0.01, 0.02), **kw), timeout=10)
        return left
    return asyncio.run(go()), calls, loaded


def test_failed_symbols_are_retried_until_they_load():
    left, calls, loaded = _run_load_all(["A", "B", "C", "D"], {"B": 3, "C": 1})
    assert left == [] and sorted(loaded) == ["A", "B", "C", "D"]
    assert calls.count("B") == 4 and calls.count("C") == 2 and calls.count("A") == 1   # only failures are repeated


def test_a_streak_of_failures_ends_the_round_instead_of_hammering_the_api():
    # B and C both fail first -> round 1 must stop before touching D, which is then loaded in round 2
    left, calls, loaded = _run_load_all(["A", "B", "C", "D"], {"B": 1, "C": 1})
    assert calls.index("D") > calls.index("C") and "D" in loaded and left == []
    assert calls[:3] == ["A", "B", "C"] and calls[3] != "D"      # D was not requested during the failing streak


def test_stop_event_ends_retrying():
    async def go():
        stop = asyncio.Event()
        n = {"calls": 0}

        async def always_fail(sym):
            n["calls"] += 1
            if n["calls"] >= 3:
                stop.set()
            raise RuntimeError("rate limited")
        return await asyncio.wait_for(history_store.load_all(["A"], always_fail, lambda *a: None, stop, asyncio.Lock(),
                                                             spacing=0, retry_delays=(0.01,)), timeout=10)
    assert asyncio.run(go()) == ["A"]
