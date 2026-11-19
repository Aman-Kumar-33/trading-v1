"""Parquet file helpers for the historical data lake.

Layout (see backtest_data/):
    raw/yfinance/<interval>/<FILE_STEM>.parquet   as downloaded from Yahoo (only merged, never edited)
    clean/<interval>/<FILE_STEM>.parquet          cleaned output

All writes are atomic: data goes to a temp file first and is then renamed over the old file,
so a crash or power cut mid-write can never leave a half-written, corrupt file behind.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd


def file_stem(yf_symbol: str) -> str:
    """Filesystem-friendly name for a Yahoo symbol.

    RELIANCE.NS -> RELIANCE, ^NSEI -> IDX_NSEI, M&M.NS -> M&M, TATAMOTORS.BO -> TATAMOTORS.BO
    """
    if yf_symbol.startswith("^"):
        return "IDX_" + yf_symbol[1:]
    if yf_symbol.endswith(".NS"):
        return yf_symbol[:-3]
    return yf_symbol


def yf_symbol_from_stem(stem: str) -> str:
    if stem.startswith("IDX_"):
        return "^" + stem[4:]
    if stem.endswith(".BO"):
        return stem
    return stem + ".NS"


def write_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp")
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False, compression="zstd")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def merge_and_write(new: pd.DataFrame, path: Path, key: str = "ts") -> pd.DataFrame:
    """Merge `new` rows into the file at `path`. Rows with the same `key` are replaced by the new
    version (Yahoo sometimes revises recent bars). Returns the merged frame."""
    old = read(path)
    if old is not None and not old.empty:
        merged = pd.concat([old, new], ignore_index=True)
    else:
        merged = new.copy()
    merged = merged.drop_duplicates(subset=[key], keep="last").sort_values(key).reset_index(drop=True)
    write_atomic(merged, path)
    return merged


def list_stems(directory: Path) -> list[str]:
    if not directory.exists():
        return []
    return sorted(p.stem for p in directory.glob("*.parquet"))
