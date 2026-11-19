"""yfinance downloader: raw OHLCV history into backtest_data/raw/yfinance/<interval>/.

How it avoids getting blocked:
  * every per-symbol request passes through a global TokenBucket (requests/minute cap)
  * rate-limit responses trigger exponential backoff; repeated ones trip a long circuit-breaker pause
  * only one downloader process can run at a time (ProcessLock)
  * the manifest records what's stored, so runs are incremental and resumable - nothing is fetched twice
  * symbols updated recently (refresh_hours) are skipped; delisted/no-data symbols are retried only weekly

Raw files hold exactly what Yahoo returned (split-adjusted OHLC, Adj Close, Volume, Dividends,
Stock Splits), minus the still-forming current bar. Cleaning happens separately in cleaner.py.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Iterable

import pandas as pd

from backend.backtest_data_pipeline.manifest import Manifest
from backend.backtest_data_pipeline.rate_limiter import RateLimitGuard, TokenBucket, is_rate_limit_error
from backend.core.clock import IST
from backend.core.config import Settings
from backend.core.timeframes import delta, is_intraday
from backend.data.market_calendar import MarketCalendar
from backend.storage import parquet_store

log = logging.getLogger(__name__)

RAW_COLUMNS = ["ts", "open", "high", "low", "close", "adj_close", "volume", "dividends", "stock_splits"]
_NO_DATA_MARKERS = ("delisted", "no data", "no price data", "pricesmissing", "tzmissing", "not found",
                    "no timezone", "quote not found", "symbol may be delisted")

# fetch function signature: (tickers, **kwargs) -> (DataFrame | None, {ticker: error_message})
FetchFn = Callable[..., tuple[pd.DataFrame | None, dict[str, str]]]


def yf_fetch(tickers: list[str], **kwargs: Any) -> tuple[pd.DataFrame | None, dict[str, str]]:
    """Default fetcher: yf.download plus the per-ticker errors yfinance records internally."""
    import yfinance as yf
    from yfinance import shared

    df = yf.download(tickers, group_by="ticker", auto_adjust=False, actions=True, threads=False,
                     progress=False, repair=False, multi_level_index=True, **kwargs)
    errors = {k.upper(): str(v) for k, v in getattr(shared, "_ERRORS", {}).items()}
    return df, errors


@dataclass
class Job:
    interval: str
    symbols: list[str]
    start: date | None          # None -> period="max"
    end: date                   # exclusive
    notes: dict[str, str] = field(default_factory=dict)   # per-symbol notes, e.g. data gaps

    def describe(self) -> str:
        rng = f"{self.start or 'max'}..{self.end}"
        return f"{self.interval} {rng} [{len(self.symbols)}] {','.join(self.symbols[:5])}{'...' if len(self.symbols) > 5 else ''}"


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
    wait_seconds: float = 0.0
    duration_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests, "bars_written": self.bars_written, "ok": len(self.ok),
            "no_data": len(self.no_data), "errors": len(self.errors), "skipped_fresh": self.skipped_fresh,
            "rate_limit_hits": self.rate_limit_hits, "breaker_trips": self.breaker_trips,
            "rate_limiter_wait_s": round(self.wait_seconds, 1), "duration_s": round(self.duration_s, 1),
        }


class YFDownloader:
    def __init__(self, settings: Settings, manifest: Manifest, limiter: TokenBucket | None = None,
                 guard: RateLimitGuard | None = None, fetch_fn: FetchFn = yf_fetch,
                 now_fn: Callable[[], datetime] | None = None, calendar: MarketCalendar | None = None):
        self.cfg = settings.section("yfinance")
        self.market = settings.section("market")
        self.paths = settings.paths
        self.manifest = manifest
        self.limiter = limiter or TokenBucket(self.cfg.get("requests_per_minute", 30), self.cfg.get("burst", 5))
        self.guard = guard or RateLimitGuard(
            backoff_base=self.cfg.get("backoff_base_seconds", 30),
            backoff_max=self.cfg.get("backoff_max_seconds", 900),
            breaker_threshold=self.cfg.get("circuit_breaker_threshold", 3),
            breaker_pause=self.cfg.get("circuit_breaker_pause_seconds", 1800),
        )
        self.fetch_fn = fetch_fn
        self.now_fn = now_fn or (lambda: datetime.now(IST))
        self.calendar = calendar or MarketCalendar.from_settings(settings)
        close_h, close_m = map(int, self.market.get("close", "15:30").split(":"))
        self._close = (close_h, close_m)

    def _now_utc_str(self) -> str:
        """The downloader's own injectable clock (self.now_fn), not wall-clock - keeps manifest
        timestamps consistent with whatever time the downloader believes it is (real in production,
        fake and fixed in tests)."""
        return self.now_fn().astimezone(timezone.utc).isoformat(timespec="seconds")

    # ------------------------------------------------------------------ planning
    def interval_cfg(self, interval: str) -> dict[str, Any]:
        try:
            return self.cfg["intervals"][interval]
        except KeyError:
            raise ValueError(f"interval {interval!r} not configured in settings.yaml yfinance.intervals") from None

    def plan_jobs(self, symbols: Iterable[str], interval: str, force: bool = False,
                  report: DownloadReport | None = None) -> list[Job]:
        icfg = self.interval_cfg(interval)
        now = self.now_fn()
        today = now.date()
        end = today + timedelta(days=1)  # yfinance end is exclusive
        max_lb = icfg.get("max_lookback_days")
        earliest = today - timedelta(days=max_lb) if max_lb else None
        overlap = timedelta(days=icfg.get("overlap_days", 1))
        refresh = timedelta(hours=icfg.get("refresh_hours", 12))
        now_utc = now.astimezone(timezone.utc)

        groups: dict[date | None, list[str]] = {}
        notes: dict[str, str] = {}
        for sym in symbols:
            st = self.manifest.get(sym, interval)
            if st and not force:
                if st.status == "ok" and st.last_success_utc and now_utc - _parse_utc(st.last_success_utc) < refresh:
                    if report:
                        report.skipped_fresh += 1
                    continue
                if st.status == "no_data" and st.last_attempt_utc and \
                        now_utc - _parse_utc(st.last_attempt_utc) < timedelta(days=7):
                    if report:
                        report.skipped_fresh += 1
                    continue
            if st and st.last_bar:
                start = pd.Timestamp(st.last_bar).date() - overlap
                if earliest and start < earliest:
                    notes[sym] = f"gap: bars between {st.last_bar[:10]} and {earliest} are no longer served by Yahoo"
                    start = earliest
            else:
                start = earliest  # None for daily -> full history
            groups.setdefault(start, []).append(sym)

        window = icfg.get("window_days")
        chunk = max(1, int(self.cfg.get("chunk_size", 5)))
        jobs: list[Job] = []
        for start, syms in sorted(groups.items(), key=lambda kv: (kv[0] is not None, kv[0] or date.min)):
            for w_start, w_end in _windows(start, end, window):
                if w_start is not None and not self._has_trading_day(w_start, w_end):
                    continue   # weekend/holiday-only window: nothing to fetch, don't waste a request
                for i in range(0, len(syms), chunk):
                    part = syms[i:i + chunk]
                    jobs.append(Job(interval, part, w_start, w_end, {s: notes[s] for s in part if s in notes}))
        return jobs

    def _has_trading_day(self, start: date, end: date) -> bool:
        d = start
        while d < end:
            if self.calendar.is_trading_day(d):
                return True
            d += timedelta(days=1)
        return False

    # ------------------------------------------------------------------ execution
    def run(self, symbols: list[str], intervals: list[str], force: bool = False,
            progress: Callable[[int, int, Job], None] | None = None) -> DownloadReport:
        report = DownloadReport()
        t0 = time.monotonic()
        for interval in intervals:
            jobs = self.plan_jobs(symbols, interval, force=force, report=report)
            log.info("planned download jobs", extra={"data": {"interval": interval, "jobs": len(jobs),
                                                              "symbols": len(symbols)}})
            for n, job in enumerate(jobs, 1):
                if progress:
                    progress(n, len(jobs), job)
                self.run_job(job, report)
        report.no_data -= report.ok       # symbol had data in another window
        report.rate_limit_hits = self.guard.total_hits
        report.breaker_trips = self.guard.breaker_trips
        report.duration_s = time.monotonic() - t0
        return report

    def run_job(self, job: Job, report: DownloadReport) -> None:
        pending = list(job.symbols)
        max_retries = int(self.cfg.get("max_retries", 5))
        attempt = 0
        while pending:
            report.wait_seconds += self.limiter.acquire(len(pending))
            report.requests += len(pending)
            kwargs: dict[str, Any] = {"interval": job.interval, "timeout": self.cfg.get("request_timeout_seconds", 30)}
            if job.start is None:
                kwargs["period"] = "max"
            else:
                kwargs["start"] = job.start.isoformat()
                kwargs["end"] = job.end.isoformat()
            t0 = time.monotonic()
            try:
                df, errors = self.fetch_fn(pending, **kwargs)
            except Exception as exc:  # noqa: BLE001 - network failures, Yahoo changes, etc.
                dur = time.monotonic() - t0
                rate_limited = is_rate_limit_error(exc)
                self.manifest.log_attempt(pending, job.interval, kwargs.get("start"), kwargs.get("end"), 0,
                                          "rate_limited" if rate_limited else "error", repr(exc), dur,
                                          now=self._now_utc_str())
                if attempt >= max_retries:
                    for s in pending:
                        status = "rate_limited" if rate_limited else "error"
                        self.manifest.record_failure(s, job.interval, status, repr(exc), now=self._now_utc_str())
                        report.errors[s] = repr(exc)
                    return
                if rate_limited:
                    self.guard.on_rate_limit(attempt)
                else:
                    self.guard.on_network_error(attempt)
                attempt += 1
                continue

            dur = time.monotonic() - t0
            errors = {k.upper(): v for k, v in (errors or {}).items()}
            retry: list[str] = []
            rows_total = 0
            for sym in pending:
                err = errors.get(sym.upper())
                if err and is_rate_limit_error(err):
                    retry.append(sym)
                    continue
                frame = _extract(df, sym, job.interval)
                if err and (frame is None or frame.empty):
                    if _is_no_data_error(err) and self._has_stored(sym, job.interval):
                        # "no price data" for a window of an already-stored symbol just means no new bars
                        rows_total += self._store(sym, job, None, report)
                        continue
                    status = "no_data" if _is_no_data_error(err) else "error"
                    self.manifest.record_failure(sym, job.interval, status, err, now=self._now_utc_str())
                    if status == "no_data":
                        report.no_data.add(sym)
                    else:
                        report.errors[sym] = err
                    continue
                rows_total += self._store(sym, job, frame, report)

            self.manifest.log_attempt(pending, job.interval, kwargs.get("start"), kwargs.get("end"), rows_total,
                                      "partial_rate_limited" if retry else "ok",
                                      "; ".join(f"{s}: {errors[s.upper()]}" for s in pending if s.upper() in errors) or None,
                                      dur, now=self._now_utc_str())
            if not retry:
                self.guard.on_success()
                return
            if attempt >= max_retries:
                for s in retry:
                    self.manifest.record_failure(s, job.interval, "rate_limited", errors.get(s.upper(), "rate limited"),
                                                 now=self._now_utc_str())
                    report.errors[s] = "rate limited (retries exhausted)"
                return
            self.guard.on_rate_limit(attempt)
            attempt += 1
            pending = retry

    # ------------------------------------------------------------------ storage
    def _has_stored(self, sym: str, interval: str) -> bool:
        return (self.paths.raw / interval / f"{parquet_store.file_stem(sym)}.parquet").exists()

    def _store(self, sym: str, job: Job, frame: pd.DataFrame | None, report: DownloadReport) -> int:
        path = self.paths.raw / job.interval / f"{parquet_store.file_stem(sym)}.parquet"
        frame = self._drop_incomplete(frame, job.interval) if frame is not None else None
        existing = parquet_store.read(path)
        if frame is None or frame.empty:
            if existing is not None and not existing.empty:
                # nothing new in this window (weekend/holiday) - still a successful check
                self.manifest.record_success(sym, job.interval, _iso(existing["ts"].iloc[0]),
                                             _iso(existing["ts"].iloc[-1]), len(existing), job.notes.get(sym),
                                             now=self._now_utc_str())
                report.ok.add(sym)
            else:
                self.manifest.record_failure(sym, job.interval, "no_data", "empty response", now=self._now_utc_str())
                report.no_data.add(sym)
            return 0
        merged = parquet_store.merge_and_write(frame, path)
        self.manifest.record_success(sym, job.interval, _iso(merged["ts"].iloc[0]), _iso(merged["ts"].iloc[-1]),
                                     len(merged), job.notes.get(sym), now=self._now_utc_str())
        report.ok.add(sym)
        report.bars_written += len(frame)
        return len(frame)

    def _drop_incomplete(self, frame: pd.DataFrame, interval: str) -> pd.DataFrame:
        """Remove the still-forming bar: Yahoo returns the current, unfinished candle during market hours."""
        if frame.empty:
            return frame
        now = self.now_fn()
        if is_intraday(interval):
            ts_ist = frame["ts"].dt.tz_convert(IST)
            session_close = ts_ist.dt.normalize() + pd.Timedelta(hours=self._close[0], minutes=self._close[1])
            bar_end = (ts_ist + delta(interval)).where(ts_ist + delta(interval) <= session_close, session_close)
            return frame.loc[bar_end <= pd.Timestamp(now)].reset_index(drop=True)
        # daily: today's bar is final only after the close + 30 min (closing session, volume settles)
        final_after = now.replace(hour=self._close[0], minute=self._close[1], second=0, microsecond=0) + timedelta(minutes=30)
        if now < final_after:
            return frame.loc[frame["ts"].dt.date < now.date()].reset_index(drop=True)
        return frame


# ---------------------------------------------------------------------- helpers
def _windows(start: date | None, end: date, window_days: int | None) -> list[tuple[date | None, date]]:
    if start is None or not window_days or (end - start).days <= window_days:
        return [(start, end)]
    out = []
    cur = start
    while cur < end:
        nxt = min(end, cur + timedelta(days=window_days))
        out.append((cur, nxt))
        cur = nxt
    return out


def _is_no_data_error(err: str) -> bool:
    low = err.lower()
    compact = low.replace(" ", "")
    return any(m in low or m.replace(" ", "") in compact for m in _NO_DATA_MARKERS)


def _extract(df: pd.DataFrame | None, sym: str, interval: str) -> pd.DataFrame | None:
    """Pull one ticker's rows out of a yf.download(group_by='ticker') frame and normalise columns."""
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        if sym not in df.columns.get_level_values(0):
            return None
        sub = df[sym].copy()
    else:
        sub = df.copy()
    sub.columns = [str(c).lower().replace(" ", "_") for c in sub.columns]
    price_cols = [c for c in ("open", "high", "low", "close") if c in sub.columns]
    if not price_cols:
        return None
    sub = sub.dropna(subset=price_cols, how="all")
    if sub.empty:
        return sub
    sub = sub.reset_index()
    sub = sub.rename(columns={sub.columns[0]: "ts"})
    ts = pd.to_datetime(sub["ts"])
    if is_intraday(interval):
        # intraday raw timestamps are stored as UTC; naive ones come in exchange time (IST)
        ts = ts.dt.tz_convert("UTC") if ts.dt.tz is not None else ts.dt.tz_localize(IST).dt.tz_convert("UTC")
    else:
        # daily raw timestamps are plain session dates without timezone
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert(IST).dt.tz_localize(None)
        ts = ts.dt.normalize()
    sub["ts"] = ts.astype("datetime64[ns, UTC]") if is_intraday(interval) else ts.astype("datetime64[ns]")
    for col in RAW_COLUMNS:
        if col not in sub.columns:
            sub[col] = 0.0 if col in ("dividends", "stock_splits") else float("nan")
    sub = sub[RAW_COLUMNS]
    for col in RAW_COLUMNS[1:]:
        sub[col] = pd.to_numeric(sub[col], errors="coerce").astype("float64")
    return sub


def _iso(ts: Any) -> str:
    return pd.Timestamp(ts).isoformat()


def _parse_utc(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
