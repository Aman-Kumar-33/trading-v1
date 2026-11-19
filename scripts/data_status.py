"""Show what's in the data lake: download manifest summary and quality summary per interval.

    python scripts/data_status.py
"""
import pandas as pd

from _bootstrap import init

from backend.backtest_data_pipeline.manifest import Manifest


def main() -> int:
    settings, _ = init("data_status")
    manifest_file = settings.paths.manifest / "manifest.db"
    if manifest_file.exists():
        m = Manifest(manifest_file)
        print("Downloads (manifest):")
        for row in m.summary():
            print(f"  {row['interval']:>4}  {row['status']:<13} {row['n']:5d} symbols  {row['bars'] or 0:>12,} bars")
        failed = [s for s in m.all() if s.status not in ("ok", None)]
        if failed:
            print(f"  {len(failed)} symbol/interval pairs not ok, e.g.:")
            for s in failed[:10]:
                print(f"    {s.symbol} {s.interval} {s.status}: {(s.error or '')[:100]}")
        m.close()
    else:
        print("No downloads yet.")

    print("\nQuality (clean data):")
    for f in sorted(settings.paths.quality_reports.glob("summary_*.csv")):
        df = pd.read_csv(f)
        interval = f.stem.replace("summary_", "")
        usable = int(df["usable"].sum()) if not df.empty else 0
        print(f"  {interval:>4}  {len(df):4d} symbols  {usable:4d} usable  median score {df['score'].median():.1f}")
        worst = df.loc[~df["usable"].astype(bool)].head(5)
        for r in worst.itertuples():
            print(f"        unusable: {r.symbol} ({r.score}) {r.reasons}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
