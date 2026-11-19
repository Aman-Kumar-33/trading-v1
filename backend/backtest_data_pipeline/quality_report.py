"""Data quality scoring and reports.

Each (symbol, interval) gets a 0-100 score from the cleaner's issue counts. Symbols scoring below
cleaning.min_quality_score are marked unusable and excluded from backtests automatically.

Outputs in backtest_data/quality_reports/:
  <interval>/<STEM>.json    full issue list for one symbol
  summary_<interval>.csv    one row per symbol - score, usable flag, key counts
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from backend.core.timeframes import is_intraday


def quality_score(issues: dict[str, Any], interval: str) -> tuple[float, list[str]]:
    """Returns (score, reasons). Penalties are capped so one problem can't dominate."""
    if issues.get("empty") or not issues.get("clean_rows"):
        return 0.0, ["no usable data"]
    raw = max(1, int(issues.get("raw_rows", 0)))
    clean = int(issues.get("clean_rows", 0))
    reasons: list[str] = []
    score = 100.0

    dropped = sum(int(issues.get(k, 0)) for k in ("nan_rows_dropped", "nonpositive_dropped", "ohlc_dropped",
                                                   "close_spikes_dropped"))
    p = min(30.0, 200.0 * dropped / raw)
    if p >= 1:
        reasons.append(f"{dropped} bad rows dropped")
    score -= p

    spikes = int(issues.get("wick_spikes_repaired", 0)) + int(issues.get("close_spikes_dropped", 0))
    p = min(15.0, 3.0 * 1000.0 * spikes / max(1, clean))
    if p >= 1:
        reasons.append(f"{spikes} bad ticks")
    score -= p

    stale = int(issues.get("stale_rows", 0))
    p = min(20.0, 100.0 * stale / max(1, clean))
    if p >= 1:
        reasons.append(f"{stale} stale bars")
    score -= p

    if is_intraday(interval):
        p = min(30.0, 100.0 * float(issues.get("avg_missing_bar_frac", 0)))
        if p >= 1:
            reasons.append(f"avg {issues.get('avg_missing_bar_frac', 0):.1%} bars missing per day")
        score -= p
        days = max(1, int(issues.get("days", 0)))
        p = min(20.0, 50.0 * int(issues.get("unusable_days", 0)) / days)
        if p >= 1:
            reasons.append(f"{issues.get('unusable_days', 0)}/{days} unusable days")
        score -= p
        if days < 5:
            score -= 20
            reasons.append(f"only {days} days of data")
    else:
        p = min(20.0, 100.0 * float(issues.get("missing_day_frac", 0)))
        if p >= 1:
            reasons.append(f"{issues.get('missing_days', 0)} missing trading days")
        score -= p
        if clean < 250:
            score -= 20
            reasons.append(f"short history ({clean} days)")
    return round(max(0.0, score), 1), reasons


def write_symbol_report(directory: Path, interval: str, stem: str, issues: dict[str, Any],
                        score: float, usable: bool, reasons: list[str]) -> None:
    out = directory / interval
    out.mkdir(parents=True, exist_ok=True)
    payload = {"symbol": stem, "interval": interval, "score": score, "usable": usable,
               "reasons": reasons, "issues": issues}
    (out / f"{stem}.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def write_summary(directory: Path, interval: str, rows: list[dict[str, Any]]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"summary_{interval}.csv"
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["usable", "score"], ascending=[True, True])
    df.to_csv(path, index=False)
    return path


def usable_symbols(directory: Path, interval: str) -> list[str]:
    """Stems that passed the quality threshold for `interval` (what backtests should load)."""
    path = directory / f"summary_{interval}.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return sorted(df.loc[df["usable"].astype(bool), "symbol"].astype(str))
