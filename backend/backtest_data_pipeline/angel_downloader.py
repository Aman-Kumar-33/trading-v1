"""Angel One SmartAPI historical downloader: raw OHLCV into backtest_data/raw/angelone/<interval>/.

Simpler than the yfinance downloader in one way (Angel's getCandleData takes one symbol per call, no
batching to chunk) but needs its own date-range chunking per interval, since a request spanning too many
days is rejected. Rather than trusting a hardcoded "maximum days" number (Angel can change this without
notice, and forum reports of undocumented behaviour are common - see the plan's Angel One section), a
job that fails with a "range too large" response is split in half and retried; the smaller window is
remembered for that interval for the rest of the run. Everything else - rate limiting, backoff, circuit
breaker, resumable manifest, atomic Parquet writes - reuses the exact same building blocks as
yf_downloader.py.

Uses its own manifest DB (backtest_data/manifest/angel_manifest.db), separate from yfinance's, since the
two sources are independent and shouldn't share bookkeeping.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

import pandas as pd

from backend.backtest_data_pipeline.manifest import Manifest
from backend.backtest_data_pipeline.rate_limiter import RateLimitGuard, TokenBucket, is_rate_limit_error
from backend.core.clock import IST
from backend.core.config import Settings
from backend.storage import parquet_store

log = logging.getLogger(__name__)

RAW_COLUMNS = ["ts", "open", "high", "low", "close", "volume"]
MIN_WINDOW_DAYS = 3   # if halving ever goes below this, give up on that job rather than loop forever

_RANGE_TOO_LARGE_MARKERS = ("date range", "days should be", "should be less than", "exceeds", "too many days",
                           "maximum", "invalid daterange", "date-range")

# (symbol_info, interval, start, end) -> (DataFrame | None, error_message | None); matches
# AngelOneAdapter.get_candle_data's signature so a real adapter or a fake can be passed interchangeably.
FetchFn = Callable[[dict[str, str], str, datetime, datetime], tuple[pd.DataFrame | None, str | None]]


def is_range_too_large_error(err: str) -> bool:
    low = err.lower()
    return any(m in low for m in _RANGE_TOO_LARGE_MARKERS)


@dataclass
class Job:
    interval: str
    symbol: str                 # our symbol, e.g. "RELIANCE" or "^NSEI"
    symbol_info: dict[str, str]  # {"token": ..., "exchange": ...}
    start: date
    end: date                   # inclusive


@dataclass
class DownloadReport:
    requests: int = 0
    bars_written: int = 0
    ok: set[str] = field(default_factory=set)
    no_data: set[str] = field(default_factory=set)
    errors: dict[str, str] = field(default_factory=dict)
    skipped_fresh: int = 0
    rate_limit_hits: int = 0
    breaker_trips: int = 0
    window_shrinks: int = 0
    wait_seconds: float = 0.0
    duration_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"requests": self.requests, "bars_written": self.bars_written, "ok": len(self.ok),
               "no_data": len(self.no_data), "errors": len(self.errors), "skipped_fresh": self.skipped_fresh,
               "rate_limit_hits": self.rate_limit_hits, "breaker_trips": self.breaker_trips,
               "window_shrinks": self.window_shrinks, "rate_limiter_wait_s": round(self.wait_seconds, 1),
               "duration_s": round(self.duration_s, 1)}


class AngelDownloader:
    def __init__(self, settings: Settings, manifest: Manifest, fetch_fn: FetchFn, limiter: TokenBucket | None = None,
                guard: RateLimitGuard | None = None, now_fn: Callable[[], datetime] | None = None):
        self.cfg = settings.section("angelone")
        self.paths = settings.paths
        self.manifest = manifest
        self.fetch_fn = fetch_fn
        self.limiter = limiter or TokenBucket(self.cfg.get("requests_per_minute", 15), self.cfg.get("burst", 2))
        self.guard = guard or RateLimitGuard(
            backoff_base=self.cfg.get("backoff_base_seconds", 30), backoff_max=self.cfg.get("backoff_max_seconds", 900),
            breaker_threshold=self.cfg.get("circuit_breaker_threshold", 3),
            breaker_pause=self.cfg.get("circuit_breaker_pause_seconds", 1800))
        self.now_fn = now_fn or (lambda: datetime.now(IST))
        self._window_days: dict[str, int] = {iv: c["window_days"] for iv, c in self.cfg.get("intervals", {}).items()}

    def interval_cfg(self, interval: str) -> dict[str, Any]:
        try:
            return self.cfg["intervals"][interval]
        except KeyError:
            raise ValueError(f"interval {interval!r} not configured in settings.yaml angelone.intervals") from None

    # ------------------------------------------------------------------ planning
    def plan_jobs(self, symbols: dict[str, dict[str, str]], interval: str, default_start: date,
                 force: bool = False, report: DownloadReport | None = None) -> list[Job]:
        icfg = self.interval_cfg(interval)
        now = self.now_fn()
        today = now.date()
        refresh = timedelta(hours=icfg.get("refresh_hours", 24))
        overlap = timedelta(days=1)
        window = max(MIN_WINDOW_DAYS, self._window_days.get(interval, icfg["window_days"]))

        jobs: list[Job] = []
        for symbol, info in symbols.items():
            st = self.manifest.get(symbol, interval)
            if st and not force and st.status == "ok" and st.last_success_utc:
                age = now.astimezone(timezone.utc) - _parse_utc(st.last_success_utc)
                if age < refresh:
                    if report:
                        report.skipped_fresh += 1
                    continue
            start = (pd.Timestamp(st.last_bar).date() - overlap) if (st and st.last_bar) else default_start
            if start >= today:
                continue
            cur = start
            while cur < today:
                nxt = min(today, cur + timedelta(days=window))
                jobs.append(Job(interval, symbol, info, cur, nxt))
                cur = nxt
        return jobs

    # ------------------------------------------------------------------ execution
    def run(self, symbols: dict[str, dict[str, str]], intervals: list[str], default_start: date,
           force: bool = False, progress: Callable[[int, int, Job], None] | None = None) -> DownloadReport:
        report = DownloadReport()
        t0 = time.monotonic()
        for interval in intervals:
            jobs = self.plan_jobs(symbols, interval, default_start, force=force, report=report)
            log.info("planned Angel One download jobs", extra={"data": {"interval": interval, "jobs": len(jobs)}})
            n = 0
            while jobs:
                job = jobs.pop(0)
                n += 1
                if progress:
                    progress(n, n + len(jobs), job)
                extra = self.run_job(job, report)
                jobs = extra + jobs   # halved sub-jobs (if any) are processed next, same interval
        report.rate_limit_hits = self.guard.total_hits
        report.breaker_trips = self.guard.breaker_trips
        report.duration_s = time.monotonic() - t0
        return report

    def run_job(self, job: Job, report: DownloadReport, max_retries: int | None = None) -> list[Job]:
        """Runs one job. Returns a (possibly empty) list of smaller replacement jobs if the range had to
        be split - the caller re-queues them."""
        max_retries = max_retries if max_retries is not None else self.cfg.get("max_retries", 6)
        start_dt = datetime.combine(job.start, datetime.min.time())
        end_dt = datetime.combine(job.end, datetime.min.time())
        attempt = 0
        while True:
            report.wait_seconds += self.limiter.acquire()
            report.requests += 1
            t0 = time.monotonic()
            try:
                df, err = self.fetch_fn(job.symbol_info, job.interval, start_dt, end_dt)
            except Exception as exc:  # noqa: BLE001 - SmartAPIException subclasses, network errors, etc.
                dur = time.monotonic() - t0
                rate_limited = is_rate_limit_error(exc)
                self.manifest.log_attempt([job.symbol], job.interval, job.start.isoformat(), job.end.isoformat(),
                                          0, "rate_limited" if rate_limited else "error", repr(exc), dur)
                if attempt >= max_retries:
                    self.manifest.record_failure(job.symbol, job.interval,
                                                 "rate_limited" if rate_limited else "error", repr(exc))
                    report.errors[job.symbol] = repr(exc)
                    return []
                (self.guard.on_rate_limit if rate_limited else self.guard.on_network_error)(attempt)
                attempt += 1
                continue

            dur = time.monotonic() - t0
            if err:
                if is_rate_limit_error(err):
                    self.manifest.log_attempt([job.symbol], job.interval, job.start.isoformat(), job.end.isoformat(),
                                              0, "rate_limited", err, dur)
                    if attempt >= max_retries:
                        self.manifest.record_failure(job.symbol, job.interval, "rate_limited", err)
                        report.errors[job.symbol] = f"rate limited (retries exhausted): {err}"
                        return []
                    self.guard.on_rate_limit(attempt)
                    attempt += 1
                    continue
                if is_range_too_large_error(err) and (job.end - job.start).days > MIN_WINDOW_DAYS:
                    return self._split_job(job, report, err)
                self.manifest.log_attempt([job.symbol], job.interval, job.start.isoformat(), job.end.isoformat(),
                                          0, "error", err, dur)
                self.manifest.record_failure(job.symbol, job.interval, "error", err)
                report.errors[job.symbol] = err
                return []

            self.guard.on_success()
            self.manifest.log_attempt([job.symbol], job.interval, job.start.isoformat(), job.end.isoformat(),
                                      len(df) if df is not None else 0, "ok", None, dur)
            self._store(job, df, report)
            return []

    def _split_job(self, job: Job, report: DownloadReport, err: str) -> list[Job]:
        span = (job.end - job.start).days
        new_window = max(MIN_WINDOW_DAYS, span // 2)
        if new_window < self._window_days.get(job.interval, new_window):
            self._window_days[job.interval] = new_window
        report.window_shrinks += 1
        log.warning("Angel One rejected the date range - splitting and shrinking the window for this interval",
                   extra={"data": {"symbol": job.symbol, "interval": job.interval, "old_span_days": span,
                                   "new_window_days": new_window, "error": err}})
        mid = job.start + timedelta(days=new_window)
        return [Job(job.interval, job.symbol, job.symbol_info, job.start, mid),
                Job(job.interval, job.symbol, job.symbol_info, mid, job.end)]

    def _store(self, job: Job, df: pd.DataFrame | None, report: DownloadReport) -> None:
        path = self.paths.raw.parent / "angelone" / job.interval / f"{parquet_store.file_stem(job.symbol)}.parquet"
        if df is None or df.empty:
            existing = parquet_store.read(path)
            if existing is not None and not existing.empty:
                self.manifest.record_success(job.symbol, job.interval, _iso(existing["ts"].iloc[0]),
                                             _iso(existing["ts"].iloc[-1]), len(existing))
                report.ok.add(job.symbol)
            else:
                self.manifest.record_failure(job.symbol, job.interval, "no_data", "empty response for this window")
                report.no_data.add(job.symbol)
            return
        merged = parquet_store.merge_and_write(df, path)
        self.manifest.record_success(job.symbol, job.interval, _iso(merged["ts"].iloc[0]), _iso(merged["ts"].iloc[-1]),
                                     len(merged))
        report.ok.add(job.symbol)
        report.bars_written += len(df)


def _iso(ts: Any) -> str:
    return pd.Timestamp(ts).isoformat()


def _parse_utc(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        from datetime import timezone
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
