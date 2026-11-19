"""Replay a historical trading day through the live market-data pipeline.

Uses clean 1m bars as the tick source, runs the real session manager + candle builder + storage, writes
the end-of-day snapshot to data/replay/<date>/, then checks that the rebuilt candles match the 1m data
resampled offline (parity check - the live pipeline must reproduce the same candles as research code).

    python scripts/replay_day.py --date 2026-09-25 --symbols RELIANCE,TCS,INFY
    python scripts/replay_day.py --date 2026-09-25 --symbols RELIANCE --speed 60   # 1 market minute per second
"""
import argparse
import asyncio
from datetime import date, timedelta

import pandas as pd

from _bootstrap import init

from backend.brokers.replay import ReplayAdapter, replay_start_time
from backend.core import events
from backend.core.clock import ReplayClock
from backend.core.events import EventBus
from backend.data.market_calendar import MarketCalendar
from backend.data.market_data import MarketDataService
from backend.data.resample import resample_bars
from backend.data.session import SessionManager
from backend.storage.db import init_db


async def replay(settings, trading_date: date, symbols: list[str], speed: float) -> int:
    calendar = MarketCalendar.from_settings(settings)
    st = calendar.session_times(trading_date)
    if st is None:
        print(f"{trading_date} is not a trading day")
        return 1
    bus = EventBus()
    clock = ReplayClock(replay_start_time(trading_date))
    conn = init_db(settings.paths.db_file)
    timeframes = settings.section("candles").get("timeframes", ["1m", "5m", "15m", "1h", "1d"])
    mds = MarketDataService(bus, calendar, conn, timeframes, settings.paths.live_data / "replay", source="replay")
    freeze_delay = int(settings.section("market").get("eod_freeze_delay_minutes", 5))
    session = SessionManager(calendar, clock, bus, freeze_delay)

    counts: dict[str, int] = {}
    bus.subscribe(events.CANDLE_CLOSED, lambda c: counts.__setitem__(c.timeframe, counts.get(c.timeframe, 0) + 1))

    adapter = ReplayAdapter(settings, bus, clock, trading_date, speed=speed)
    await adapter.subscribe(symbols)
    await adapter.connect()
    if not adapter.available_symbols():
        print("no clean 1m data for that date - download (1m) and clean it first")
        return 1
    await session.step()
    sent = await adapter.run(on_time=[session.step, mds.on_clock],
                             finish_at=st.close + timedelta(minutes=freeze_delay + 1))

    print(f"ticks replayed: {sent}   session state: {session.state.value}   frozen: {mds.frozen}")
    print("candles closed:", dict(sorted(counts.items())))
    print("builder stats:", mds.builder.stats)

    # parity check vs offline resampling
    bad = 0
    for sym in adapter.available_symbols():
        bars = adapter.day_bars(sym)
        for tf in timeframes:
            expected = resample_bars(bars, tf)
            built = pd.DataFrame([c.to_dict() for c in mds.builder.today(sym, tf)])
            if built.empty or len(built) != len(expected):
                print(f"  PARITY FAIL {sym} {tf}: {len(built)} built vs {len(expected)} expected")
                bad += 1
                continue
            cols = ["open", "high", "low", "close", "volume"]
            diff = (built[cols].to_numpy(dtype=float) - expected[cols].to_numpy(dtype=float))
            if abs(diff).max() > 1e-6 or not (built["start"].to_numpy() == expected["ts"].to_numpy()).all():
                print(f"  PARITY FAIL {sym} {tf}: values differ (max abs diff {abs(diff).max():.6f})")
                bad += 1
    print("parity check:", "OK - live candles match offline resampling" if bad == 0 else f"{bad} mismatches")
    conn.close()
    return 0 if bad == 0 else 3


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--symbols", required=True, help="comma separated NSE symbols")
    p.add_argument("--speed", type=float, default=0.0, help="0 = as fast as possible, 60 = 1 min per second")
    args = p.parse_args()
    settings, _ = init("replay_day")
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    return asyncio.run(replay(settings, date.fromisoformat(args.date), symbols, args.speed))


if __name__ == "__main__":
    raise SystemExit(main())
