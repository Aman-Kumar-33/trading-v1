"""Clean raw data into backtest_data/clean/ and write quality reports.

    python scripts/clean_data.py                        # all intervals, all downloaded symbols
    python scripts/clean_data.py --intervals 1d,5m
    python scripts/clean_data.py --symbols RELIANCE,IDX_NSEI
"""
import argparse

from _bootstrap import init

from backend.backtest_data_pipeline.pipeline import clean_all
from backend.storage.parquet_store import file_stem


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--intervals", default="1d,1h,15m,5m,1m")
    p.add_argument("--symbols", help="comma separated file stems (IDX_NSEI) or NSE symbols (RELIANCE)")
    args = p.parse_args()
    settings, log = init("clean_data")
    intervals = [i.strip() for i in args.intervals.split(",") if i.strip()]
    stems = None
    if args.symbols:
        stems = [s.strip() if s.strip().startswith("IDX_") else file_stem(s.strip().upper() + ".NS")
                 for s in args.symbols.split(",") if s.strip()]

    def progress(interval, n, total, stem):
        if n == 1 or n % 50 == 0 or n == total:
            print(f"[{interval}] {n}/{total} {stem}", flush=True)

    report = clean_all(settings, intervals, stems, progress=progress)
    for interval, d in report.per_interval.items():
        print(f"  {interval:>4}: {d['symbols']} symbols, {d['usable']} usable, {d['unusable']} below quality threshold")
    print(f"  reports: {settings.paths.quality_reports}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
