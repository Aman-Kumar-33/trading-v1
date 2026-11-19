"""End-to-end proof of Phase 4: replay a real historical trading day through the FULL live pipeline,
now including the pieces that previously didn't exist - StrategyRunner, Allocator, RiskManager,
OrderManager, PaperBroker. Nothing before this script ever actually called a strategy from live candle
events; this is the first time the whole chain (ticks -> candles -> features -> strategy -> allocator ->
risk manager -> paper broker -> trade) runs together.

This intentionally runs a strategy that was proven unprofitable (orb) for the proof itself - the goal
here is mechanical correctness (does the pipeline wire together and produce sane, costed trades), not
finding profit. Don't read anything about strategy quality into this script's P&L.

    python scripts/paper_trade_replay.py --date 2026-09-25 --symbols RELIANCE,TCS --strategy orb
"""
import argparse
import asyncio
import json
from datetime import date, timedelta

from _bootstrap import init

from backend.brokers.paper import PaperBroker
from backend.brokers.replay import ReplayAdapter, replay_start_time
from backend.core import events
from backend.core.clock import ReplayClock
from backend.core.events import EventBus
from backend.core.models import Mode
from backend.data.market_calendar import MarketCalendar
from backend.data.market_data import MarketDataService
from backend.data.session import SessionManager
from backend.ensemble.allocator import Allocator
from backend.ensemble.performance_tracker import PerformanceTracker
from backend.ensemble.shadow_book import ShadowBook
from backend.execution.order_manager import OrderManager
from backend.execution.strategy_runner import StrategyRunner
from backend.risk.risk_manager import RiskLimits, RiskManager
from backend.storage import parquet_store
from backend.storage.db import init_db
from backend.strategies import INTRADAY_STRATEGIES
from backend.strategies.registry import build_strategy


def load_history(settings, symbol: str, timeframe: str, n_bars: int):
    path = settings.paths.clean / timeframe / f"{parquet_store.file_stem(symbol)}.parquet"
    df = parquet_store.read(path)
    if df is None or df.empty:
        return None
    if "day_ok" in df.columns:
        df = df.loc[df["day_ok"]].reset_index(drop=True)
    return df.tail(n_bars).reset_index(drop=True)


async def run(settings, trading_date: date, symbols: list[str], strategy_id: str, timeframe: str,
             capital: float) -> int:
    calendar = MarketCalendar.from_settings(settings)
    st = calendar.session_times(trading_date)
    if st is None:
        print(f"{trading_date} is not a trading day")
        return 1

    bus = EventBus()
    clock = ReplayClock(replay_start_time(trading_date))
    conn = init_db(settings.paths.db_file)
    timeframes = settings.section("candles").get("timeframes", ["1m", "5m", "15m", "1h", "1d"])
    mds = MarketDataService(bus, calendar, conn, timeframes, settings.paths.live_data / "replay",
                            source="replay-paper")
    freeze_delay = int(settings.section("market").get("eod_freeze_delay_minutes", 5))
    session = SessionManager(calendar, clock, bus, freeze_delay)

    cls = next(c for c in INTRADAY_STRATEGIES if c.strategy_id == strategy_id)
    strat = build_strategy(cls, "v1")
    history = {(sym, timeframe): load_history(settings, sym, timeframe, 3000) for sym in symbols}
    if timeframe != "1d":   # daily bars feed the intraday daily-trend filter (completed days only)
        history.update({(sym, "1d"): load_history(settings, sym, "1d", 1000) for sym in symbols})
    history = {k: v for k, v in history.items() if v is not None and not v.empty}

    broker = PaperBroker(bus, starting_cash=capital)
    await broker.connect()
    risk_manager = RiskManager(RiskLimits(capital=capital))
    tracker = PerformanceTracker()
    order_manager = OrderManager(broker, risk_manager, mode=Mode.PAPER)
    shadow_book = ShadowBook(tracker, risk_manager)
    allocator = Allocator(tracker=tracker)
    runner = StrategyRunner(bus, strategies_by_timeframe={timeframe: [strat]},
                            allowed_strategy_ids={strategy_id}, history_by_symbol_tf=history,
                            allocator=allocator, risk_manager=risk_manager, order_manager=order_manager,
                            shadow_book=shadow_book)

    candle_counts: dict[str, int] = {}
    bus.subscribe(events.CANDLE_CLOSED, lambda c: candle_counts.__setitem__(c.timeframe, candle_counts.get(c.timeframe, 0) + 1))

    adapter = ReplayAdapter(settings, bus, clock, trading_date, speed=0.0)
    await adapter.subscribe(symbols)
    await adapter.connect()
    if not adapter.available_symbols():
        print("no clean 1m data for that date - download and clean it first")
        return 1
    await session.step()
    # order_manager.on_time is ALSO called per-tick by StrategyRunner.on_tick, but that alone isn't
    # enough: clean intraday data deliberately stops at 15:14 (the cleaner drops bars from 15:15 on -
    # see README "Yahoo Finance facts"), so replayed ticks never reach the 15:15 square-off cutoff on
    # their own. ReplayAdapter.run()'s `finish_at` mechanism exists exactly for this - one final hook
    # call past the last real tick - so it must be in this list too, or intraday positions never close.
    async def shadow_on_time(now):
        shadow_book.on_time(now)

    sent = await adapter.run(on_time=[session.step, mds.on_clock, order_manager.on_time, shadow_on_time],
                             finish_at=st.close + timedelta(minutes=freeze_delay + 1))

    print(f"ticks replayed: {sent}   candles closed: {candle_counts}")
    print(f"open positions remaining: {len(order_manager.open_positions)} (should be 0 - square-off should have closed everything)")
    print(f"trades: {len(order_manager.closed_trades)}")
    for t in order_manager.closed_trades:
        print(f"  {t.position.symbol:10s} {t.position.side.value:4s} qty={t.position.qty:5d} "
             f"entry={t.position.entry_price:8.2f} exit={t.exit_price:8.2f} reason={t.exit_reason:10s} "
             f"pnl_net={t.pnl_net:9.2f}")
    net_pnl = sum(t.pnl_net for t in order_manager.closed_trades)
    print(f"net P&L: {net_pnl:.2f}   kill switch tripped: {risk_manager.state.kill_switch.tripped}")
    print(f"shadow trades scored for the ensemble: {shadow_book.recorded}")
    conn.close()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--symbols", required=True, help="comma separated NSE symbols")
    p.add_argument("--strategy", default="orb")
    p.add_argument("--timeframe", default="5m")
    p.add_argument("--capital", type=float, default=1_000_000.0, help="placeholder paper capital")
    args = p.parse_args()
    settings, _ = init("paper_trade_replay")
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    return asyncio.run(run(settings, date.fromisoformat(args.date), symbols, args.strategy, args.timeframe,
                          args.capital))


if __name__ == "__main__":
    raise SystemExit(main())
