"""Splits, bonuses and dividends.

Yahoo's `Close` is split-adjusted (and bonus-adjusted - a 1:1 bonus appears as a 2.0 split), `Adj Close`
is additionally dividend-adjusted. For realistic backtests we need three price views:
  * adjusted (split + dividend)  -> indicators and returns (no fake gaps on ex-dates)
  * split-adjusted (Yahoo Close) -> continuous price series
  * unadjusted (actual traded)   -> order prices, circuit limits, position sizing at the time

Unadjusted price on date d = split-adjusted price x product of split ratios that happened *after* d.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backend.storage import parquet_store

ACTION_COLUMNS = ["date", "dividend", "split_ratio"]


def extract_actions(raw_daily: pd.DataFrame) -> pd.DataFrame:
    """Rows of a raw daily frame that carry a dividend or split, as (date, dividend, split_ratio)."""
    if raw_daily is None or raw_daily.empty:
        return pd.DataFrame(columns=ACTION_COLUMNS)
    df = raw_daily[["ts", "dividends", "stock_splits"]].copy()
    df["dividends"] = df["dividends"].fillna(0.0)
    df["stock_splits"] = df["stock_splits"].fillna(0.0)
    df = df.loc[(df["dividends"] != 0) | (df["stock_splits"] != 0)]
    dates = pd.to_datetime(df["ts"])
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    out = pd.DataFrame({
        "date": dates.dt.normalize(),
        "dividend": df["dividends"].astype(float),
        "split_ratio": df["stock_splits"].replace(0.0, 1.0).astype(float),
    })
    return out.reset_index(drop=True)


def save_actions(actions: pd.DataFrame, directory: Path, stem: str) -> Path:
    path = directory / f"{stem}.parquet"
    parquet_store.write_atomic(actions, path)
    return path


def load_actions(directory: Path, stem: str) -> pd.DataFrame:
    df = parquet_store.read(directory / f"{stem}.parquet")
    return df if df is not None else pd.DataFrame(columns=ACTION_COLUMNS)


def split_factor_after(dates: pd.Series, actions: pd.DataFrame) -> pd.Series:
    """For each date, the product of split ratios strictly after it (1.0 if none).

    `dates` may be naive or tz-aware; only the calendar date matters.
    """
    d = pd.to_datetime(dates)
    if d.dt.tz is not None:
        d = d.dt.tz_localize(None)
    d = d.dt.normalize()
    splits = actions.loc[actions["split_ratio"].astype(float) != 1.0] if not actions.empty else actions
    if splits.empty:
        return pd.Series(np.ones(len(d)), index=dates.index)
    split_dates = pd.to_datetime(splits["date"]).dt.normalize().to_numpy()
    ratios = splits["split_ratio"].astype(float).to_numpy()
    order = np.argsort(split_dates)
    split_dates, ratios = split_dates[order], ratios[order]
    # suffix products: factor for "after index i"
    suffix = np.append(np.cumprod(ratios[::-1])[::-1], 1.0)
    idx = np.searchsorted(split_dates, d.to_numpy(), side="right")  # first split strictly after the date
    return pd.Series(suffix[idx], index=dates.index)
