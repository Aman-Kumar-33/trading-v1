"""Download historical OHLCV from Angel One SmartAPI into backtest_data/raw/angelone/ (rate-limited,
resumable, self-adjusting date windows - see backend/backtest_data_pipeline/angel_downloader.py).

Needs Angel One credentials in .env (copy .env.example) - see the plan's "Angel One SmartAPI setup"
section for exactly what to fill in on Angel's site and where to find these:
    ANGEL_API_KEY, ANGEL_CLIENT_CODE, ANGEL_PIN, ANGEL_TOTP_SECRET

Examples:
    python scripts/fetch_angel_history.py --dry-run                      # plan only, no login, no requests
    python scripts/fetch_angel_history.py --symbols RELIANCE,TCS --intervals 1h --years-back 1
    python scripts/fetch_angel_history.py                                # default: NIFTY100 + key indices,
                                                                          # 5m/15m/1h, 3 years back

Safe to stop (Ctrl+C) and re-run - it continues where it left off. Designed to run for a long time
(hours) across the full default universe; that's expected, not a bug - it's not something to watch
turn-by-turn, let it run in the background and check logs/fetch_angel_history.jsonl if anything failed.
"""
import argparse
from datetime import date, timedelta

from _bootstrap import init

from backend.backtest_data_pipeline.angel_downloader import AngelDownloader
from backend.backtest_data_pipeline.manifest import Manifest
from backend.backtest_data_pipeline.rate_limiter import LockHeld, ProcessLock
from backend.brokers.angel import AngelLoginError, AngelOneAdapter
from backend.core.events import EventBus
from backend.core.secrets import MissingCredentials, get_angel_credentials
from backend.data import angel_instruments as ai
from backend.data.universe import load_universe

DEFAULT_INTERVALS = ["5m", "15m", "1h"]
# Angel's own names for the indices we track (verified against the real instrument master 2026-09-29 -
# see backend/data/angel_instruments.py). Angel doesn't have a fixed naming convention, so this is an
# explicit lookup, not a pattern match.
INDEX_NAMES = {
    "^NSEI": "Nifty 50", "^NSEBANK": "Nifty Bank", "^CNXIT": "Nifty IT", "^CNXPHARMA": "Nifty Pharma",
    "^CNX100": "Nifty 100", "^CRSLDX": "Nifty 500", "^NSMIDCP": "Nifty Next 50",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--universe", default="nifty100", help="index list name (default: nifty100)")
    p.add_argument("--symbols", help="comma separated NSE symbols (overrides --universe)")
    p.add_argument("--no-indices", action="store_true")
    p.add_argument("--intervals", default=",".join(DEFAULT_INTERVALS))
    p.add_argument("--years-back", type=float, default=3.0, help="history depth for a first-time fetch")
    p.add_argument("--limit", type=int, help="only the first N symbols (for testing)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--refresh-master", action="store_true", help="force-refresh the instrument master file")
    p.add_argument("--dry-run", action="store_true", help="show the plan, log in, but make no data requests")
    return p.parse_args()


def build_symbol_universe(settings, ai_master, args) -> dict[str, dict[str, str]]:
    eq_map = ai.build_symbol_map(ai_master, exchange="NSE")
    if args.symbols:
        wanted = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        wanted = load_universe(settings, args.universe)
    if args.limit:
        wanted = wanted[: args.limit]
    out: dict[str, dict[str, str]] = {}
    missing = []
    for sym in wanted:
        info = eq_map.get(sym)
        if info:
            out[sym] = {"token": info["token"], "exchange": info["exchange"]}
        else:
            missing.append(sym)
    if missing:
        print(f"  {len(missing)} symbols not found in Angel's instrument master (skipped): {missing[:10]}"
             + (" ..." if len(missing) > 10 else ""))
    if not args.no_indices:
        idx_map = ai.build_index_map(ai_master, INDEX_NAMES)
        for name, info in idx_map.items():
            out[name] = {"token": info["token"], "exchange": info["exchange"]}
    return out


def main() -> int:
    args = parse_args()
    settings, log = init("fetch_angel_history")
    intervals = [i.strip() for i in args.intervals.split(",") if i.strip()]
    default_start = date.today() - timedelta(days=int(args.years_back * 365.25))

    master_cfg = settings.section("angelone")
    master_path = settings.paths.universe / "angel_instrument_master.json"
    master = ai.load_instrument_master(master_path, master_cfg["instrument_master_url"],
                                       master_cfg.get("instrument_master_refresh_hours", 24),
                                       force=args.refresh_master)
    symbols = build_symbol_universe(settings, master, args)
    print(f"  universe: {len(symbols)} instruments, intervals: {intervals}, history from {default_start}")

    if args.dry_run:
        manifest = Manifest(settings.paths.manifest / "angel_manifest.db")
        dl = AngelDownloader(settings, manifest, fetch_fn=lambda *a, **k: (None, None))
        total = 0
        for interval in intervals:
            jobs = dl.plan_jobs(symbols, interval, default_start, force=args.force)
            total += len(jobs)
            print(f"  {interval:>4}: {len(jobs):4d} requests planned")
            for j in jobs[:2]:
                print(f"        e.g. {j.symbol} {j.start}..{j.end}")
        rpm = master_cfg.get("requests_per_minute", 15)
        print(f"  total ~{total} requests -> at least {total / rpm:.0f} min at {rpm}/min (no login performed)")
        manifest.close()
        return 0

    try:
        creds = get_angel_credentials()
    except MissingCredentials as exc:
        log.error(str(exc))
        print(f"ERROR: {exc}")
        return 2

    bus = EventBus()
    adapter = AngelOneAdapter(bus, creds.api_key, creds.client_code, creds.pin, creds.totp_secret)
    try:
        adapter.login()
    except AngelLoginError as exc:
        log.error("Angel One login failed", extra={"data": {"error": str(exc)}})
        print(f"LOGIN FAILED: {exc}\n(check ANGEL_CLIENT_CODE/ANGEL_PIN/ANGEL_TOTP_SECRET in .env, and that "
             f"your app is approved on Angel One's developer portal)")
        return 3

    manifest = Manifest(settings.paths.manifest / "angel_manifest.db")
    lock = ProcessLock(settings.paths.manifest / "angel_downloader.lock",
                       master_cfg.get("lock_stale_hours", 12))

    def progress(n, total, job):
        if n == 1 or n % 20 == 0 or n == total:
            print(f"[{job.interval}] request {n}/{total}: {job.symbol} {job.start}..{job.end}", flush=True)

    try:
        with lock:
            dl = AngelDownloader(settings, manifest, fetch_fn=adapter.get_candle_data)
            report = dl.run(symbols, intervals, default_start, force=args.force, progress=progress)
    except LockHeld as exc:
        log.error(str(exc))
        return 2
    except KeyboardInterrupt:
        print("\nstopped - progress is saved, re-run to continue")
        return 130
    finally:
        manifest.close()

    log.info("Angel One download finished", extra={"data": report.as_dict()})
    for k, v in report.as_dict().items():
        print(f"  {k:22s} {v}")
    if report.errors:
        print("  errors (first 10):")
        for s, e in list(report.errors.items())[:10]:
            print(f"    {s}: {e[:150]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
