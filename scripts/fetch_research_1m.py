"""Download 1-minute candles for the NIFTY 50 from Angel One, for testing the live-MACD tolerance on many days.

Angel has no historical ticks, but it serves 1-minute bars (at least a year back, up to 8,000 per request), which is
the finest history there is. Files are written to research_data/angel_1m/<SYMBOL>/<window start>.parquet, one per
25-calendar-day window, so the download can be stopped and restarted - finished windows are skipped. Newest first.

    python scripts/fetch_research_1m.py --from 2026-04-01
"""
import argparse
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.brokers.angel import AngelOneAdapter  # noqa: E402
from backend.core.clock import IST  # noqa: E402
from backend.core.config import load_settings  # noqa: E402
from backend.core.events import EventBus  # noqa: E402
from backend.core.secrets import get_angel_credentials  # noqa: E402
from backend.data import angel_instruments as ai  # noqa: E402
from backend.data import trending  # noqa: E402

WINDOW_DAYS = 25          # <= ~18 sessions x 375 bars = 6,750 rows, under Angel's 8,000-row cap
SPACING = 3.5             # seconds between requests (Angel allows ~15 a minute)
RETRY_WAITS = (10, 30, 60, 120)


def windows(start: date, end: date):
    out, a = [], start
    while a <= end:
        out.append((a, min(a + timedelta(days=WINDOW_DAYS - 1), end)))
        a += timedelta(days=WINDOW_DAYS)
    return list(reversed(out))   # newest first


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="2026-04-01")
    ap.add_argument("--to", dest="end", default=str(date.today()))
    ap.add_argument("--symbols", help="comma separated; default the whole NIFTY 50")
    args = ap.parse_args()
    settings = load_settings()
    out_root = settings.paths.root / "research_data" / "angel_1m"
    creds = get_angel_credentials()
    angel = AngelOneAdapter(EventBus(), creds.api_key, creds.client_code, creds.pin, creds.totp_secret)
    angel.login()
    mc = settings.section("angelone")
    master = ai.load_instrument_master(settings.paths.universe / "angel_instrument_master.json", mc["instrument_master_url"], 24)
    symbol_map = ai.build_symbol_map(master, exchange="NSE")
    symbols = args.symbols.split(",") if args.symbols else trending.universe_symbols(settings)
    missing = [s for s in symbols if s not in symbol_map]
    symbols = [s for s in symbols if s in symbol_map]
    print(f"{len(symbols)} symbols, not in the instrument master: {missing}", flush=True)
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    todo = [(w, s) for w in windows(start, end) for s in symbols]
    done = failed = 0
    for (a, b), sym in todo:
        path = out_root / sym.replace("&", "_AND_") / f"{a}.parquet"
        if path.exists():
            done += 1
            continue
        s = datetime(a.year, a.month, a.day, 9, 0, tzinfo=IST)
        e = datetime(b.year, b.month, b.day, 15, 35, tzinfo=IST)
        df = None
        for wait in (0, *RETRY_WAITS):
            time.sleep(wait or SPACING)
            try:
                df, err = angel.get_candle_data(symbol_map[sym], "1m", s, e)
            except Exception as exc:  # noqa: BLE001 - Angel's rate limit arrives as a non-JSON body
                err = repr(exc)[:120]
            if err is None:
                break
            df = None
        if df is None:
            failed += 1
            print(f"FAILED {sym} {a}: {err}", flush=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        done += 1
        if done % 10 == 0:
            print(f"{done}/{len(todo)} done, {failed} failed; last {sym} {a} rows={len(df)}", flush=True)
    print(f"FINISHED: {done}/{len(todo)} done, {failed} failed", flush=True)
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
