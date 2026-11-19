"""Download historical OHLCV from Yahoo Finance into backtest_data/raw/ (rate-limited, resumable).

Examples:
    python scripts/download_history.py                          # default universe, all intervals
    python scripts/download_history.py --intervals 1d           # daily only
    python scripts/download_history.py --symbols RELIANCE,TCS --intervals 1d,5m
    python scripts/download_history.py --universe nifty50 --dry-run   # show the plan, download nothing
    python scripts/download_history.py --force                  # ignore refresh_hours, re-check everything

Safe to stop (Ctrl+C) and re-run at any time - it continues where it left off.
"""
import argparse

from _bootstrap import init

from backend.backtest_data_pipeline.manifest import Manifest
from backend.backtest_data_pipeline.rate_limiter import LockHeld, ProcessLock
from backend.backtest_data_pipeline.yf_downloader import YFDownloader
from backend.data.universe import download_symbols, yf_symbol

ALL_INTERVALS = ["1d", "1h", "15m", "5m", "1m"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--universe", help="index list name (default from settings.yaml)")
    p.add_argument("--symbols", help="comma separated NSE symbols (overrides --universe)")
    p.add_argument("--no-indices", action="store_true", help="skip the index tickers")
    p.add_argument("--intervals", default=",".join(ALL_INTERVALS))
    p.add_argument("--limit", type=int, help="only the first N symbols (for testing)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def resolve_symbols(settings, args) -> list[str]:
    if args.symbols:
        symbols = [yf_symbol(s.strip().upper()) for s in args.symbols.split(",") if s.strip()]
        if not args.no_indices:
            symbols = list(settings.section("universe").get("indices", [])) + symbols
    else:
        symbols = download_symbols(settings, args.universe, include_indices=not args.no_indices)
    return symbols[: args.limit] if args.limit else symbols


def main() -> int:
    args = parse_args()
    settings, log = init("download_history")
    symbols = resolve_symbols(settings, args)
    intervals = [i.strip() for i in args.intervals.split(",") if i.strip()]

    manifest = Manifest(settings.paths.manifest / "manifest.db")
    downloader = YFDownloader(settings, manifest)

    if args.dry_run:
        rpm = settings.section("yfinance").get("requests_per_minute", 30)
        total = 0
        for interval in intervals:
            jobs = downloader.plan_jobs(symbols, interval, force=args.force)
            reqs = sum(len(j.symbols) for j in jobs)
            total += reqs
            print(f"{interval:>4}: {len(jobs):5d} jobs, {reqs:6d} symbol-requests")
            for j in jobs[:3]:
                print(f"      e.g. {j.describe()}")
        print(f"total ~{total} requests -> at least {total / rpm:.0f} min at {rpm}/min")
        manifest.close()
        return 0

    lock = ProcessLock(settings.paths.manifest / "downloader.lock",
                       settings.section("yfinance").get("lock_stale_hours", 12))

    def progress(n, total, job):
        if n == 1 or n % 10 == 0 or n == total:
            print(f"[{job.interval}] job {n}/{total}: {job.describe()}", flush=True)

    try:
        with lock:
            report = downloader.run(symbols, intervals, force=args.force, progress=progress)
    except LockHeld as exc:
        log.error(str(exc))
        return 2
    except KeyboardInterrupt:
        print("\nstopped - progress is saved, re-run to continue")
        return 130
    finally:
        manifest.close()

    log.info("download finished", extra={"data": report.as_dict()})
    for k, v in report.as_dict().items():
        print(f"  {k:22s} {v}")
    if report.errors:
        print("  errors (first 10):")
        for s, e in list(report.errors.items())[:10]:
            print(f"    {s}: {e[:150]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
