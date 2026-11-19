"""On-disk cache for build_features output. Building features on 3 years of 5m bars takes ~2 minutes per
symbol (chart-pattern pivot scanning dominates) and every research script used to redo it on every run.

The cache key covers everything the output depends on: the raw bars (count, first/last timestamp, a
checksum of the closes), the daily bars passed in (for daily_trend), the timeframe, and a hash of the
feature code itself - edit anything under backend/features/ and every cached file is invalidated
automatically. Cache files live in backtest_data/features_cache/<timeframe>/; stale ones for the same
symbol are deleted when a new one is written.
"""
from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd

from backend.features.pipeline import build_features
from backend.storage import parquet_store

log = logging.getLogger(__name__)

FEATURES_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def feature_code_hash() -> str:
    h = hashlib.sha256()
    for p in sorted(FEATURES_DIR.rglob("*.py")):
        h.update(p.relative_to(FEATURES_DIR).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def _frame_fingerprint(df: pd.DataFrame | None) -> str:
    if df is None or df.empty:
        return "none"
    closes = pd.to_numeric(df["close"], errors="coerce").fillna(0.0)
    return f"{len(df)}|{df['ts'].iloc[0]}|{df['ts'].iloc[-1]}|{float(closes.sum()):.6f}"


def cache_key(raw: pd.DataFrame, timeframe: str, daily: pd.DataFrame | None) -> str:
    blob = "|".join([feature_code_hash(), timeframe, _frame_fingerprint(raw), _frame_fingerprint(daily)])
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def cached_build_features(cache_root: Path, symbol: str, raw: pd.DataFrame, timeframe: str,
                          daily: pd.DataFrame | None = None) -> pd.DataFrame:
    """build_features(raw, timeframe, daily=daily), read from / written to the cache."""
    stem = parquet_store.file_stem(symbol)
    folder = Path(cache_root) / timeframe
    path = folder / f"{stem}__{cache_key(raw, timeframe, daily)}.parquet"
    if path.exists():
        return _restore_text_columns(pd.read_parquet(path))
    feat = build_features(raw, timeframe, daily=daily)
    folder.mkdir(parents=True, exist_ok=True)
    for stale in folder.glob(f"{stem}__*.parquet"):
        stale.unlink(missing_ok=True)
    parquet_store.write_atomic(feat, path)
    return feat


def _restore_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Parquet reads text columns (regime, chart_pattern, ...) back as pandas string dtype with NaN for
    missing; a fresh build_features has object dtype with None. Restore that exactly."""
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) and not pd.api.types.is_object_dtype(df[col]):
            s = df[col].astype(object)
            df[col] = s.where(s.notna(), None)
    return df
