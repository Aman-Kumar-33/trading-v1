"""Raw -> clean data cleaning.

Rules (in order):
  1. timestamps to IST; duplicates removed; sorted
  2. rows with missing prices dropped
  3. intraday: only regular-session bars (09:15-15:30) kept
  4. non-trading days removed (weekends, known holidays, days the reference index didn't trade -
     unless the stock actually traded volume that day, which is kept and counted as "unverified")
  5. OHLC sanity: prices must be > 0, high >= max(open, close), low <= min(open, close);
     tiny violations are repaired, big ones dropped
  6. bad ticks: wicks far beyond the neighbourhood are clipped to the body; closes that jump away by more
     than N ATR and come straight back are dropped
  7. stale feed: runs of identical bars with no volume are flagged
  8. gaps: missing bars are counted, NEVER filled with invented prices; intraday days missing too many
     bars (or disagreeing with the daily bar) are marked day_ok = False
  9. corporate actions: adjusted and unadjusted price columns added

Nothing is silently changed: every repair/drop is counted in the returned `issues` dict, which feeds
the quality report.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, time
from typing import Any

import numpy as np
import pandas as pd

from backend.backtest_data_pipeline.corporate_actions import split_factor_after
from backend.core.clock import IST
from backend.core.timeframes import bars_per_session, is_intraday, minutes


@dataclass
class CleanResult:
    df: pd.DataFrame
    issues: dict[str, Any] = field(default_factory=dict)


class DataCleaner:
    def __init__(self, cleaning_cfg: dict[str, Any], market_cfg: dict[str, Any]):
        self.cfg = cleaning_cfg
        oh, om = map(int, market_cfg.get("open", "09:15").split(":"))
        ch, cm = map(int, market_cfg.get("close", "15:30").split(":"))
        self.open_t, self.close_t = time(oh, om), time(ch, cm)
        eh, em = map(int, str(cleaning_cfg.get("intraday_data_end", market_cfg.get("close", "15:30"))).split(":"))
        self.data_end_t = min(self.close_t, time(eh, em))

    def clean(self, raw: pd.DataFrame, interval: str, *, trading_days: set[date] | None = None,
              holidays: set[date] | None = None, actions: pd.DataFrame | None = None,
              daily_ref: pd.DataFrame | None = None, is_index: bool = False) -> CleanResult:
        intraday = is_intraday(interval)
        issues: dict[str, Any] = {"raw_rows": 0 if raw is None else len(raw)}
        if raw is None or raw.empty:
            issues["empty"] = True
            return CleanResult(pd.DataFrame(), issues)

        df = raw.copy()
        # 1. timestamps
        ts = pd.to_datetime(df["ts"])
        if intraday:
            ts = ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts
            ts = ts.dt.tz_convert(IST)
        else:
            ts = ts.dt.tz_localize(None) if ts.dt.tz is not None else ts
            ts = ts.dt.normalize().dt.tz_localize(IST)
        df["ts"] = ts
        before = len(df)
        df = df.drop_duplicates(subset=["ts"], keep="last").sort_values("ts").reset_index(drop=True)
        issues["duplicates_dropped"] = before - len(df)

        # 2. missing prices
        before = len(df)
        df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
        issues["nan_rows_dropped"] = before - len(df)
        df["volume"] = df["volume"].fillna(0.0)

        # 3. session filter
        if intraday:
            t = df["ts"].dt.time
            in_session = (t >= self.open_t) & (t < self.close_t)
            reliable = t < self.data_end_t
            issues["out_of_session_dropped"] = int((~in_session).sum())
            issues["after_data_end_dropped"] = int((in_session & ~reliable).sum())
            df = df.loc[in_session & reliable].reset_index(drop=True)

        # 4. non-trading days
        d = df["ts"].dt.date
        weekend = df["ts"].dt.weekday >= 5
        known_holiday = d.isin(holidays) if holidays else pd.Series(False, index=df.index)
        if trading_days:
            # the reference index only proves anything inside the date range it covers
            ref_lo, ref_hi = min(trading_days), max(trading_days)
            not_ref_day = ~d.isin(trading_days) & (d >= ref_lo) & (d <= ref_hi)
        else:
            not_ref_day = pd.Series(False, index=df.index)
        has_volume = df["volume"] > 0
        drop = weekend | known_holiday | (not_ref_day & ~has_volume)
        issues["non_trading_day_dropped"] = int(drop.sum())
        issues["unverified_day_rows"] = int((not_ref_day & has_volume & ~weekend & ~known_holiday).sum())
        df = df.loc[~drop].reset_index(drop=True)

        # 5. OHLC sanity
        before = len(df)
        df = df.loc[(df[["open", "high", "low", "close"]] > 0).all(axis=1) & (df["volume"] >= 0)].reset_index(drop=True)
        issues["nonpositive_dropped"] = before - len(df)
        df, fixed, dropped = self._fix_ohlc(df)
        issues["ohlc_repaired"], issues["ohlc_dropped"] = fixed, dropped

        # 6. bad ticks
        df, wick_fixes, spike_drops = self._spikes(df)
        issues["wick_spikes_repaired"], issues["close_spikes_dropped"] = wick_fixes, spike_drops

        # 7. stale runs
        run_len = int(self.cfg.get("stale_run_intraday" if intraday else "stale_run_daily", 10 if intraday else 3))
        df["flag_stale"] = self._stale_mask(df, run_len, ignore_volume=is_index)
        issues["stale_rows"] = int(df["flag_stale"].sum())
        df["flag_zero_volume"] = (df["volume"] == 0) if not is_index else False
        issues["zero_volume_rows"] = int(df["flag_zero_volume"].sum())

        # 8. gaps
        if intraday:
            df, gap_issues = self._intraday_days(df, interval, daily_ref, is_index)
        else:
            gap_issues = self._daily_gaps(df, trading_days)
        issues.update(gap_issues)

        # 9. corporate actions / price views
        df = self._price_views(df, intraday, actions)

        df["volume"] = df["volume"].round().astype("int64")
        issues["clean_rows"] = len(df)
        if len(df):
            issues["first_bar"] = df["ts"].iloc[0].isoformat()
            issues["last_bar"] = df["ts"].iloc[-1].isoformat()
        return CleanResult(df, issues)

    # ------------------------------------------------------------------ steps
    def _fix_ohlc(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
        tol = float(self.cfg.get("ohlc_fix_tolerance", 0.001))
        body_hi = df[["open", "close"]].max(axis=1)
        body_lo = df[["open", "close"]].min(axis=1)
        hi_viol = (body_hi - df["high"]).clip(lower=0) / body_hi
        lo_viol = (df["low"] - body_lo).clip(lower=0) / body_lo
        hl_viol = (df["low"] - df["high"]).clip(lower=0) / df["high"]
        bad = (hi_viol > 0) | (lo_viol > 0) | (hl_viol > 0)
        big = (hi_viol > tol) | (lo_viol > tol) | (hl_viol > tol)
        fixable = bad & ~big
        df = df.loc[~big].copy()
        fixable = fixable.loc[df.index]
        df.loc[fixable, "high"] = df.loc[fixable, ["open", "high", "low", "close"]].max(axis=1)
        df.loc[fixable, "low"] = df.loc[fixable, ["open", "high", "low", "close"]].min(axis=1)
        df["flag_repaired"] = fixable
        return df.reset_index(drop=True), int(fixable.sum()), int(big.sum())

    def _atr(self, df: pd.DataFrame) -> pd.Series:
        """Robust ATR: rolling median of true range, using only past bars."""
        prev_c = df["close"].shift(1)
        tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_c).abs(), (df["low"] - prev_c).abs()], axis=1).max(axis=1)
        w = int(self.cfg.get("atr_window", 20))
        atr = tr.rolling(w, min_periods=5).median().shift(1)
        atr = atr.fillna(tr.expanding(min_periods=1).median())
        # floor: tiny ATRs (flat stretches) would make every normal move look like a spike
        floor = df["close"] * 0.001
        return atr.where(atr > floor, floor)

    def _spikes(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
        if len(df) < 3:
            df["flag_spike"] = False
            return df, 0, 0
        k = float(self.cfg.get("spike_atr_mult", 8))
        atr = self._atr(df)
        body_hi = df[["open", "close"]].max(axis=1)
        body_lo = df[["open", "close"]].min(axis=1)
        nb_high = pd.concat([df["high"].shift(1), df["high"].shift(-1)], axis=1).max(axis=1)
        nb_low = pd.concat([df["low"].shift(1), df["low"].shift(-1)], axis=1).min(axis=1)

        # wick spikes: wick far beyond body AND far beyond both neighbours -> clip wick to body
        hi_spike = ((df["high"] - body_hi) > k * atr) & ((df["high"] - nb_high) > 0.5 * k * atr)
        lo_spike = ((body_lo - df["low"]) > k * atr) & ((nb_low - df["low"]) > 0.5 * k * atr)
        df = df.copy()
        df.loc[hi_spike, "high"] = body_hi[hi_spike]
        df.loc[lo_spike, "low"] = body_lo[lo_spike]
        wick_fixes = int(hi_spike.sum() + lo_spike.sum())

        # close spikes: jumps away from previous close and straight back at the next close
        prev_c, next_c = df["close"].shift(1), df["close"].shift(-1)
        up = ((df["close"] - prev_c) > k * atr) & ((df["close"] - next_c) > k * atr)
        down = ((prev_c - df["close"]) > k * atr) & ((next_c - df["close"]) > k * atr)
        close_spike = (up | down).fillna(False)
        df["flag_spike"] = hi_spike | lo_spike
        df = df.loc[~close_spike].reset_index(drop=True)
        return df, wick_fixes, int(close_spike.sum())

    @staticmethod
    def _stale_mask(df: pd.DataFrame, run_len: int, ignore_volume: bool) -> pd.Series:
        if df.empty:
            return pd.Series(False, index=df.index)
        same = (df[["open", "high", "low", "close"]].diff().abs().sum(axis=1) == 0)
        cond = same if ignore_volume else same & (df["volume"] == 0)
        # a bar starts a new run whenever the condition breaks
        grp = (~cond).cumsum()
        run = cond.groupby(grp).transform("sum")
        # include the first bar of the run (the one the duplicates are equal to)
        start_of_run = cond.shift(-1, fill_value=False) & ~cond
        mask = (cond & (run + 1 >= run_len))
        mask = mask | (start_of_run & mask.shift(-1, fill_value=False))
        return mask.astype(bool)

    def expected_bars(self, interval: str) -> int:
        """Bars a complete day should have, up to the reliable data end (intraday_data_end)."""
        span = (self.data_end_t.hour * 60 + self.data_end_t.minute) - (self.open_t.hour * 60 + self.open_t.minute)
        return min(bars_per_session(interval), -(-span // minutes(interval)))

    def _intraday_days(self, df: pd.DataFrame, interval: str, daily_ref: pd.DataFrame | None,
                       is_index: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
        if df.empty:
            df["day_ok"] = pd.Series(dtype=bool)
            return df, {"days": 0}
        expected = self.expected_bars(interval)
        day = df["ts"].dt.date
        counts = day.value_counts()
        missing_frac = (1 - counts / expected).clip(lower=0)
        max_missing = float(self.cfg.get("max_missing_bar_frac", 0.10))
        bad_days = set(missing_frac[missing_frac > max_missing].index)

        mismatch_days: set[date] = set()
        if daily_ref is not None and not daily_ref.empty:
            tol = float(self.cfg.get("crosscheck_hl_tolerance", 0.02))
            agg = df.groupby(day).agg(high=("high", "max"), low=("low", "min"))
            ref = daily_ref.assign(d=daily_ref["ts"].dt.date).set_index("d")[["high", "low"]]
            j = agg.join(ref, rsuffix="_d", how="inner")
            # only complete days are comparable
            j = j.loc[~j.index.isin(bad_days)]
            bad = ((j["high"] - j["high_d"]).abs() / j["high_d"] > tol) | ((j["low"] - j["low_d"]).abs() / j["low_d"] > tol)
            mismatch_days = set(j.index[bad])
        df["day_ok"] = ~day.isin(bad_days | mismatch_days)
        return df, {
            "days": int(len(counts)),
            "expected_bars_per_day": expected,
            "avg_missing_bar_frac": round(float(missing_frac.mean()), 4),
            "incomplete_days": len(bad_days),
            "daily_mismatch_days": len(mismatch_days),
            "unusable_days": len(bad_days | mismatch_days),
        }

    @staticmethod
    def _daily_gaps(df: pd.DataFrame, trading_days: set[date] | None) -> dict[str, Any]:
        if df.empty or not trading_days:
            return {"missing_days": 0}
        have = set(df["ts"].dt.date)
        first, last = min(have), max(have)
        expected = {d for d in trading_days if first <= d <= last}
        missing = expected - have
        return {"missing_days": len(missing), "missing_day_frac": round(len(missing) / max(1, len(expected)), 4)}

    @staticmethod
    def _price_views(df: pd.DataFrame, intraday: bool, actions: pd.DataFrame | None) -> pd.DataFrame:
        if df.empty:
            return df
        factor = split_factor_after(df["ts"], actions) if actions is not None and not actions.empty \
            else pd.Series(np.ones(len(df)), index=df.index)
        df["close_unadj"] = df["close"] * factor
        if not intraday:
            adj = df["adj_close"].where(df["adj_close"].notna() & (df["adj_close"] > 0), df["close"])
            df["adj_factor"] = adj / df["close"]
            for c in ("open", "high", "low", "close"):
                df[f"{c}_adj"] = df[c] * df["adj_factor"]
        cols = [c for c in df.columns if c not in ("dividends", "stock_splits", "adj_close")]
        return df[cols]


def trading_days_from_reference(ref_raw_daily: pd.DataFrame | None, holidays: set[date],
                                extend_to: date | None = None) -> set[date]:
    """Historical trading days = days the reference index (NIFTY 50) has a bar, plus weekdays after its
    last bar that aren't known holidays (so the most recent days are never lost)."""
    days: set[date] = set()
    last = None
    if ref_raw_daily is not None and not ref_raw_daily.empty:
        ts = pd.to_datetime(ref_raw_daily["ts"])
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert(IST).dt.tz_localize(None)
        days = set(ts.dt.date)
        last = max(days)
    if extend_to and last:
        for d in pd.bdate_range(last, extend_to).date:
            if d not in holidays:
                days.add(d)
    return days
