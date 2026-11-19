"""Daily data collector - run once every trading day after market close (e.g. 16:30 IST).

Steps:
  1. refresh NSE holiday list (ignored if NSE is unreachable)
  2. refresh index lists, F&O ban list, price bands
  3. incremental download of all intervals (manifest + refresh_hours decide what's actually fetched)
  4. clean the downloaded data and rebuild quality reports

Because Yahoo keeps only ~30 days of 1m and ~60 days of 5m/15m bars, running this every day is what
builds our own long intraday history.

    python scripts/daily_collect.py
    python scripts/daily_collect.py --skip-clean
"""
import argparse
from datetime import datetime

from _bootstrap import init

from backend.backtest_data_pipeline.manifest import Manifest
from backend.backtest_data_pipeline.pipeline import clean_all
from backend.backtest_data_pipeline.rate_limiter import LockHeld, ProcessLock
from backend.backtest_data_pipeline.yf_downloader import YFDownloader
from backend.core.clock import IST
from backend.data.market_calendar import MarketCalendar, refresh_holidays_file
from backend.data.nse_client import NSEClient
from backend.data.universe import download_symbols, refresh_index_lists, refresh_market_lists, sync_instruments
from backend.storage.db import init_db

INTERVALS = ["1d", "1h", "15m", "5m", "1m"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--universe", help="index list name (default from settings.yaml)")
    p.add_argument("--skip-clean", action="store_true")
    args = p.parse_args()
    settings, log = init("daily_collect")

    now = datetime.now(IST)
    calendar = MarketCalendar.from_settings(settings)
    if calendar.is_trading_day(now.date()) and now.time() < calendar.close_t:
        log.warning("market is still open - today's bars are incomplete and will be skipped until the next run")

    client = NSEClient()
    try:
        path = settings.resolve(settings.section("market").get("holidays_file", "config/nse_holidays.yaml"))
        refresh_holidays_file(path, client.holidays())
    except Exception as exc:  # noqa: BLE001
        log.warning("holiday refresh failed - using cached list", extra={"data": {"error": repr(exc)}})
    counts = refresh_index_lists(settings, client)
    counts.update(refresh_market_lists(settings, client))
    conn = init_db(settings.paths.db_file)
    counts["instruments"] = sync_instruments(conn, settings, args.universe)
    conn.close()
    log.info("universe refreshed", extra={"data": counts})

    symbols = download_symbols(settings, args.universe)
    manifest = Manifest(settings.paths.manifest / "manifest.db")
    lock = ProcessLock(settings.paths.manifest / "downloader.lock",
                       settings.section("yfinance").get("lock_stale_hours", 12))
    try:
        with lock:
            report = YFDownloader(settings, manifest).run(symbols, INTERVALS)
    except LockHeld as exc:
        log.error(str(exc))
        return 2
    finally:
        manifest.close()
    log.info("download finished", extra={"data": report.as_dict()})
    print("download:", report.as_dict())

    if not args.skip_clean:
        clean_report = clean_all(settings, INTERVALS)
        print("clean:", clean_report.per_interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
