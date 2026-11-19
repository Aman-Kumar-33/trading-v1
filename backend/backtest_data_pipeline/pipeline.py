"""Cleaning pipeline: raw/ -> corporate_actions/ + clean/ + quality_reports/.

Order matters:
  1. corporate actions from raw daily bars
  2. trading-day calendar from the reference index (NIFTY 50) + holiday list
  3. clean daily bars (needed to cross-check intraday bars)
  4. clean intraday intervals, cross-checked against the clean daily bars
  5. quality score + reports for every (symbol, interval)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

import pandas as pd

from backend.backtest_data_pipeline import corporate_actions as ca
from backend.backtest_data_pipeline.cleaner import DataCleaner, trading_days_from_reference
from backend.backtest_data_pipeline.quality_report import quality_score, write_summary, write_symbol_report
from backend.core.config import Settings
from backend.data.market_calendar import MarketCalendar
from backend.storage import parquet_store

log = logging.getLogger(__name__)

INTERVAL_ORDER = ["1d", "1h", "15m", "5m", "1m"]
# Angel One is only fetched for these intervals (yfinance's own daily/1m history is already deep enough
# not to need it) - see backend/backtest_data_pipeline/angel_downloader.py.
ANGEL_INTERVALS = {"5m", "15m", "1h"}


def merge_raw_sources(*frames: pd.DataFrame | None) -> pd.DataFrame | None:
    """Combine raw bars from more than one source (e.g. yfinance + Angel One) for the same symbol/
    interval into one frame, keyed by timestamp. Timestamps are normalized to UTC first (matching the
    same "naive means UTC" assumption `DataCleaner.clean` already makes) so a tz-naive yfinance frame and
    a tz-aware Angel One frame concatenate correctly instead of producing a mixed-dtype column. On an
    overlapping timestamp the later frame in `frames` wins - callers pass the deeper/preferred source last."""
    present = [f for f in frames if f is not None and not f.empty]
    if not present:
        return None
    normalized = []
    for f in present:
        f = f.copy()
        f["ts"] = pd.to_datetime(f["ts"], utc=True)
        normalized.append(f)
    merged = pd.concat(normalized, ignore_index=True)
    return merged.drop_duplicates(subset=["ts"], keep="last").sort_values("ts").reset_index(drop=True)


@dataclass
class CleanReport:
    per_interval: dict[str, dict[str, int]] = field(default_factory=dict)

    def add(self, interval: str, usable: bool) -> None:
        d = self.per_interval.setdefault(interval, {"symbols": 0, "usable": 0, "unusable": 0})
        d["symbols"] += 1
        d["usable" if usable else "unusable"] += 1


def clean_all(settings: Settings, intervals: list[str], stems: list[str] | None = None,
              progress: Callable[[str, int, int, str], None] | None = None) -> CleanReport:
    paths = settings.paths
    cleaner = DataCleaner(settings.section("cleaning"), settings.section("market"))
    min_score = float(settings.section("cleaning").get("min_quality_score", 70))
    calendar = MarketCalendar.from_settings(settings)
    holidays = calendar.holiday_set()
    report = CleanReport()

    ref_stem = parquet_store.file_stem(settings.section("universe").get("calendar_reference", "^NSEI"))
    ref_raw = parquet_store.read(paths.raw / "1d" / f"{ref_stem}.parquet")
    if ref_raw is None:
        log.warning("reference index daily data missing - trading-day check limited to weekends/holidays",
                    extra={"data": {"reference": ref_stem}})
    trading_days = trading_days_from_reference(ref_raw, holidays, extend_to=date.today())

    ordered = [i for i in INTERVAL_ORDER if i in intervals] + [i for i in intervals if i not in INTERVAL_ORDER]
    angel_root = paths.raw.parent / "angelone"
    # daily must be cleaned first if any intraday interval is requested (cross-check source)
    for interval in ordered:
        raw_dir = paths.raw / interval
        angel_dir = angel_root / interval if interval in ANGEL_INTERVALS else None
        if stems is not None:
            todo = stems
        else:
            todo = set(parquet_store.list_stems(raw_dir))
            if angel_dir is not None:
                todo |= set(parquet_store.list_stems(angel_dir))
            todo = sorted(todo)
        rows = []
        for n, stem in enumerate(todo, 1):
            if progress:
                progress(interval, n, len(todo), stem)
            raw = parquet_store.read(raw_dir / f"{stem}.parquet")
            if angel_dir is not None:
                # Angel One's history goes back years vs. yfinance's ~60 days for 5m/15m and ~730 for
                # 1h - merge it in so the deeper history is actually used, not left on disk unused.
                raw = merge_raw_sources(raw, parquet_store.read(angel_dir / f"{stem}.parquet"))
            if raw is None:
                continue
            is_index = stem.startswith("IDX_")
            if interval == "1d":
                actions = ca.extract_actions(raw)
                ca.save_actions(actions, paths.corporate_actions, stem)
            else:
                actions = ca.load_actions(paths.corporate_actions, stem)
            daily_ref = None
            if interval != "1d":
                daily_ref = parquet_store.read(paths.clean / "1d" / f"{stem}.parquet")
            try:
                result = cleaner.clean(raw, interval, trading_days=trading_days, holidays=holidays,
                                       actions=actions, daily_ref=daily_ref, is_index=is_index)
            except Exception as exc:  # noqa: BLE001 - one broken file must not stop the batch
                log.exception("cleaning failed", extra={"data": {"symbol": stem, "interval": interval}})
                result_issues = {"error": repr(exc), "raw_rows": len(raw)}
                write_symbol_report(paths.quality_reports, interval, stem, result_issues, 0.0, False, ["cleaning error"])
                rows.append({"symbol": stem, "score": 0.0, "usable": False, "reasons": "cleaning error"})
                report.add(interval, False)
                continue
            score, reasons = quality_score(result.issues, interval)
            usable = score >= min_score
            if not result.df.empty:
                parquet_store.write_atomic(result.df, paths.clean / interval / f"{stem}.parquet")
            write_symbol_report(paths.quality_reports, interval, stem, result.issues, score, usable, reasons)
            rows.append({
                "symbol": stem, "score": score, "usable": usable, "reasons": "; ".join(reasons),
                "clean_rows": result.issues.get("clean_rows", 0), "first_bar": result.issues.get("first_bar"),
                "last_bar": result.issues.get("last_bar"),
                "unusable_days": result.issues.get("unusable_days", result.issues.get("missing_days", 0)),
            })
            report.add(interval, usable)
        if rows:
            write_summary(paths.quality_reports, interval, rows)
        log.info("cleaned interval", extra={"data": {"interval": interval, **report.per_interval.get(interval, {})}})
    return report


def load_clean(settings: Settings, stem: str, interval: str, only_ok_days: bool = True) -> pd.DataFrame | None:
    """Convenience loader for research code: clean bars, optionally without unusable intraday days."""
    df = parquet_store.read(settings.paths.clean / interval / f"{stem}.parquet")
    if df is not None and only_ok_days and "day_ok" in df.columns:
        df = df.loc[df["day_ok"]].reset_index(drop=True)
    return df
