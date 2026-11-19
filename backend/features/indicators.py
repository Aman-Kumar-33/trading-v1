"""Technical indicators - vectorized (whole-history-at-once) so backtests over years of data are fast.

Every function takes a DataFrame with at least `close` (and `high`/`low`/`volume` where needed) sorted by
time, and returns a Series or DataFrame aligned to the same index. Nothing here looks at future bars:
every indicator at row i uses only rows <= i, so the same code is safe for both live use and backtesting.

Session-aware indicators (VWAP, opening range, volume profile) need a `day` grouping key - pass the
`ts` column (tz-aware) and they derive the trading day themselves.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------- moving averages
def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window, min_periods=window).mean()


def ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False, min_periods=span).mean()


# ---------------------------------------------------------------------- oscillators
def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = _wilder_smooth(gain, window)
    avg_loss = _wilder_smooth(loss, window)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(avg_loss != 0, 100.0).mask(avg_gain.isna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": macd_line - signal_line})


def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 14) -> pd.Series:
    """Money Flow Index - volume-weighted RSI, 0-100."""
    typical = (high + low + close) / 3
    raw_flow = typical * volume
    rising = typical.diff() > 0
    pos_flow = raw_flow.where(rising, 0.0).rolling(window, min_periods=window).sum()
    neg_flow = raw_flow.where(~rising, 0.0).rolling(window, min_periods=window).sum()
    ratio = pos_flow / neg_flow.replace(0, np.nan)
    out = 100 - 100 / (1 + ratio)
    return out.where(neg_flow != 0, 100.0).mask(pos_flow.isna())


def bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = sma(close, window)
    std = close.rolling(window, min_periods=window).std(ddof=0)
    return pd.DataFrame({"mid": mid, "upper": mid + num_std * std, "lower": mid - num_std * std,
                         "bandwidth": (4 * num_std * std) / mid})


# ---------------------------------------------------------------------- volatility / trend
def _wilder_smooth(series: pd.Series, window: int) -> pd.Series:
    """Wilder's smoothing = EMA with alpha = 1/window, seeded by the first `window` values' mean."""
    seed = series.rolling(window, min_periods=window).mean()
    out = seed.copy()
    alpha = 1.0 / window
    vals = series.to_numpy(dtype=float)
    seed_vals = seed.to_numpy(dtype=float)
    result = np.full(len(series), np.nan)
    start = None
    for i in range(len(series)):
        if start is None:
            if not np.isnan(seed_vals[i]):
                result[i] = seed_vals[i]
                start = i
        else:
            result[i] = result[i - 1] * (1 - alpha) + vals[i] * alpha
    return pd.Series(result, index=series.index)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True range. NaN at the first bar (no previous close to compare against) - matches TA-Lib's
    TRANGE convention, rather than silently falling back to a plain high-low range there."""
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.mask(prev_close.isna())


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's ATR - the standard trading indicator (EMA-style smoothing of true range).

    Not the same as the cleaner's robust median-ATR (backend/backtest_data_pipeline/cleaner.py), which
    is deliberately outlier-resistant for data-quality checks rather than reactive for trading signals.
    """
    return _wilder_smooth(true_range(high, low, close), window)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.DataFrame:
    """Average Directional Index with +DI/-DI. adx > 25 conventionally means "trending"."""
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)
    # first bar has no previous bar to compare against - mask it (like true_range) so the Wilder seed
    # window doesn't silently include a spurious 0.0 in place of an undefined reading
    plus_dm, minus_dm = plus_dm.mask(up_move.isna()), minus_dm.mask(up_move.isna())
    tr_smooth = _wilder_smooth(true_range(high, low, close), window)
    plus_di = 100 * _wilder_smooth(plus_dm, window) / tr_smooth.replace(0, np.nan)
    minus_di = 100 * _wilder_smooth(minus_dm, window) / tr_smooth.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": _wilder_smooth(dx, window)})


def supertrend(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 10,
              multiplier: float = 3.0) -> pd.DataFrame:
    """Supertrend: a trailing stop/trend line. direction = +1 (uptrend, line below price) or -1."""
    band_atr = atr(high, low, close, window)
    hl2 = (high + low) / 2
    basic_upper = hl2 + multiplier * band_atr
    basic_lower = hl2 - multiplier * band_atr

    n = len(close)
    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    line = np.full(n, np.nan)
    direction = np.full(n, 0)
    c = close.to_numpy(dtype=float)
    bu, bl = basic_upper.to_numpy(dtype=float), basic_lower.to_numpy(dtype=float)

    start = window  # first usable ATR value
    if start >= n:
        return pd.DataFrame({"line": line, "direction": direction}, index=close.index)
    final_upper[start], final_lower[start] = bu[start], bl[start]
    direction[start] = 1 if c[start] > final_upper[start] else -1
    line[start] = final_lower[start] if direction[start] == 1 else final_upper[start]

    for i in range(start + 1, n):
        final_upper[i] = bu[i] if (bu[i] < final_upper[i - 1] or c[i - 1] > final_upper[i - 1]) else final_upper[i - 1]
        final_lower[i] = bl[i] if (bl[i] > final_lower[i - 1] or c[i - 1] < final_lower[i - 1]) else final_lower[i - 1]
        if direction[i - 1] == 1:
            direction[i] = -1 if c[i] < final_lower[i] else 1
        else:
            direction[i] = 1 if c[i] > final_upper[i] else -1
        line[i] = final_lower[i] if direction[i] == 1 else final_upper[i]

    return pd.DataFrame({"line": line, "direction": direction}, index=close.index)


# ---------------------------------------------------------------------- session-aware (intraday only)
def _session_day(ts: pd.Series) -> pd.Series:
    return ts.dt.date


def session_vwap(ts: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Volume-weighted average price, resetting at the start of each trading day."""
    typical = (high + low + close) / 3
    pv = typical * volume
    day = _session_day(ts)
    cum_pv = pv.groupby(day).cumsum()
    cum_vol = volume.groupby(day).cumsum()
    return cum_pv / cum_vol.replace(0, np.nan)


def opening_range(ts: pd.Series, high: pd.Series, low: pd.Series, minutes: int = 15) -> pd.DataFrame:
    """Opening range high/low for the first `minutes` of each session, held constant for the rest of the day
    (available to later bars only - the range for a day is only known once the window has closed)."""
    day = _session_day(ts)
    session_start = ts.groupby(day).transform("min")
    in_range = ts < (session_start + pd.Timedelta(minutes=minutes))
    or_high = high.where(in_range).groupby(day).cummax()
    or_low = low.where(in_range).groupby(day).cummin()
    or_high = or_high.groupby(day).ffill()
    or_low = or_low.groupby(day).ffill()
    window_end = session_start + pd.Timedelta(minutes=minutes)
    established = ts >= window_end
    return pd.DataFrame({"or_high": or_high.where(established), "or_low": or_low.where(established),
                         "or_established": established})


def volume_profile(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
                   day: pd.Series | None = None, bins: int = 24) -> pd.DataFrame:
    """Point of Control (POC, the price level with the most traded volume), Value Area High/Low (the
    range containing 70% of volume) for each session (or for the whole series if `day` is None).

    Returns one row per group with columns poc, vah, val (for a rolling/backtest use, call once per
    completed day and look up the previous day's levels - don't use today's own volume profile until
    today is over).
    """
    key = day if day is not None else pd.Series(0, index=high.index)
    rows = []
    for g, idx in high.groupby(key).groups.items():
        h, l, c, v = high.loc[idx], low.loc[idx], close.loc[idx], volume.loc[idx]
        lo, hi = l.min(), h.max()
        if hi <= lo or v.sum() == 0:
            rows.append({"group": g, "poc": c.iloc[-1] if len(c) else np.nan, "vah": hi, "val": lo})
            continue
        edges = np.linspace(lo, hi, bins + 1)
        mid = (h + l) / 2
        idx_bin = np.clip(np.digitize(mid, edges) - 1, 0, bins - 1)
        vol_by_bin = np.bincount(idx_bin, weights=v.to_numpy(), minlength=bins)
        poc_bin = int(np.argmax(vol_by_bin))
        poc = (edges[poc_bin] + edges[poc_bin + 1]) / 2
        order = np.argsort(-vol_by_bin)
        total = vol_by_bin.sum()
        target = 0.70 * total
        chosen = np.zeros(bins, dtype=bool)
        running = 0.0
        for b in order:
            chosen[b] = True
            running += vol_by_bin[b]
            if running >= target:
                break
        chosen_bins = np.where(chosen)[0]
        vah = edges[chosen_bins.max() + 1]
        val = edges[chosen_bins.min()]
        rows.append({"group": g, "poc": poc, "vah": vah, "val": val})
    return pd.DataFrame(rows).set_index("group")


# ---------------------------------------------------------------------- convenience
def add_core_indicators(df: pd.DataFrame, *, intraday: bool = False) -> pd.DataFrame:
    """Attach the commonly-used indicator set as new columns. `df` needs ts, open, high, low, close,
    volume. Returns a new DataFrame (input is not modified)."""
    out = df.copy()
    out["ema_9"] = ema(out["close"], 9)
    out["ema_20"] = ema(out["close"], 20)
    out["ema_50"] = ema(out["close"], 50)
    out["ema_200"] = ema(out["close"], 200)
    out["rsi_14"] = rsi(out["close"], 14)
    m = macd(out["close"])
    out["macd"], out["macd_signal"], out["macd_hist"] = m["macd"], m["signal"], m["hist"]
    out["mfi_14"] = mfi(out["high"], out["low"], out["close"], out["volume"], 14)
    bb = bollinger_bands(out["close"])
    out["bb_mid"], out["bb_upper"], out["bb_lower"], out["bb_width"] = bb["mid"], bb["upper"], bb["lower"], bb["bandwidth"]
    out["atr_14"] = atr(out["high"], out["low"], out["close"], 14)
    a = adx(out["high"], out["low"], out["close"], 14)
    out["plus_di"], out["minus_di"], out["adx_14"] = a["plus_di"], a["minus_di"], a["adx"]
    st = supertrend(out["high"], out["low"], out["close"])
    out["supertrend"], out["supertrend_dir"] = st["line"], st["direction"]
    if intraday:
        out["vwap"] = session_vwap(out["ts"], out["high"], out["low"], out["close"], out["volume"])
        orb = opening_range(out["ts"], out["high"], out["low"])
        out["or_high"], out["or_low"], out["or_established"] = orb["or_high"], orb["or_low"], orb["or_established"]
    return out
