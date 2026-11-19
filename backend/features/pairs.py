"""Pairs & lead-lag: cointegration (not just correlation), the tradeable spread and its z-score, the
half-life of mean reversion, and lead-lag cross-correlation for index/sector -> stock relationships.

Cointegration matters more than correlation for pairs trading: two stocks can be highly correlated day
to day yet drift apart forever (no mean-reverting spread to trade), or have modest correlation but a
genuinely stationary spread. `engle_granger` is the standard two-step test for this.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


@dataclass
class CointegrationResult:
    p_value: float
    t_stat: float
    hedge_ratio: float     # y ~= hedge_ratio * x + intercept
    intercept: float
    is_cointegrated: bool  # p_value below the caller's chosen significance level


def hedge_ratio(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    """OLS slope/intercept of y on x (the hedge ratio for the spread y - slope*x)."""
    x_arr = np.column_stack([np.ones(len(x)), x.to_numpy()])
    coefs, *_ = np.linalg.lstsq(x_arr, y.to_numpy(), rcond=None)
    intercept, slope = coefs
    return float(slope), float(intercept)


def engle_granger(y: pd.Series, x: pd.Series, significance: float = 0.05) -> CointegrationResult:
    """Engle-Granger two-step cointegration test between two (log-)price series. Use log prices so the
    hedge ratio is scale-free across stocks at very different price levels."""
    aligned = pd.concat([y, x], axis=1).dropna()
    t_stat, p_value, _ = coint(aligned.iloc[:, 0], aligned.iloc[:, 1])
    slope, intercept = hedge_ratio(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return CointegrationResult(p_value=float(p_value), t_stat=float(t_stat), hedge_ratio=slope,
                              intercept=intercept, is_cointegrated=p_value < significance)


def spread(y: pd.Series, x: pd.Series, hedge: float, intercept: float = 0.0) -> pd.Series:
    """The stationary combination the pairs strategy actually trades: y - (hedge*x + intercept)."""
    return y - (hedge * x + intercept)


def rolling_zscore(series: pd.Series, window: int = 60) -> pd.Series:
    """Rolling z-score - no lookahead, each bar's z-score only uses its own trailing window."""
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=0)
    return (series - mean) / std.replace(0, np.nan)


def half_life(series: pd.Series) -> float:
    """Half-life (in bars) of mean reversion, from an AR(1)/Ornstein-Uhlenbeck fit on the series' first
    difference vs its own lagged level: delta[t] = theta*(level[t-1]) + const. theta<0 means the series
    pulls back toward its mean; half-life = -ln(2)/theta. Returns inf if theta >= 0 (no mean reversion
    detected - a random walk or trending series)."""
    level = series.dropna()
    delta = level.diff().dropna()
    lagged = level.shift(1).dropna()
    aligned = pd.concat([delta, lagged], axis=1).dropna()
    aligned.columns = ["delta", "lagged"]
    x = np.column_stack([np.ones(len(aligned)), aligned["lagged"].to_numpy()])
    coefs, *_ = np.linalg.lstsq(x, aligned["delta"].to_numpy(), rcond=None)
    theta = coefs[1]
    if theta >= 0:
        return float("inf")
    return float(-np.log(2) / theta)


def scan_pairs(prices: dict[str, pd.Series], significance: float = 0.05, min_half_life: float = 1.0,
              max_half_life: float = 60.0) -> pd.DataFrame:
    """Nightly scan: test every combination of symbols in `prices` (log close series, same index) for
    cointegration, keep the ones that pass with a *tradeable* half-life (not too fast to trade around
    costs, not so slow the capital sits idle for months). Returns a DataFrame sorted by p-value
    (strongest first): symbol_y, symbol_x, p_value, hedge_ratio, half_life."""
    names = list(prices)
    rows = []
    for i in range(len(names)):
        for j in range(len(names)):
            if i == j:
                continue
            y_name, x_name = names[i], names[j]
            y, x = prices[y_name], prices[x_name]
            aligned = pd.concat([y, x], axis=1).dropna()
            if len(aligned) < 60:
                continue
            result = engle_granger(aligned.iloc[:, 0], aligned.iloc[:, 1], significance)
            if not result.is_cointegrated:
                continue
            sp = spread(aligned.iloc[:, 0], aligned.iloc[:, 1], result.hedge_ratio, result.intercept)
            hl = half_life(sp)
            if not (min_half_life <= hl <= max_half_life):
                continue
            rows.append({"symbol_y": y_name, "symbol_x": x_name, "p_value": result.p_value,
                        "hedge_ratio": result.hedge_ratio, "intercept": result.intercept, "half_life": hl})
    df = pd.DataFrame(rows, columns=["symbol_y", "symbol_x", "p_value", "hedge_ratio", "intercept", "half_life"])
    return df.sort_values("p_value").reset_index(drop=True) if not df.empty else df


# ---------------------------------------------------------------------- lead-lag
def lead_lag_correlation(leader: pd.Series, follower: pd.Series, max_lag: int = 5) -> pd.DataFrame:
    """Correlation between `leader`'s returns and `follower`'s returns shifted by each lag in
    [-max_lag, max_lag]. A positive lag k with the highest correlation means the follower's move k bars
    later tracks the leader's move now - i.e. the leader leads by k bars. Lookahead-safe to *compute*
    (it's an offline/nightly diagnostic over historical data, not a live per-bar feature); a strategy
    only uses the resulting best-lag as a fixed parameter going forward, not to peek at future bars."""
    r_lead = leader.pct_change()
    r_foll = follower.pct_change()
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        shifted = r_foll.shift(-lag)   # positive lag: follower's future return vs leader's return now
        aligned = pd.concat([r_lead, shifted], axis=1).dropna()
        corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1]) if len(aligned) > 2 else np.nan
        rows.append({"lag": lag, "correlation": corr})
    return pd.DataFrame(rows)


def best_lead_lag(leader: pd.Series, follower: pd.Series, max_lag: int = 5) -> tuple[int, float]:
    """(lag, correlation) of the strongest relationship found by `lead_lag_correlation`."""
    table = lead_lag_correlation(leader, follower, max_lag).dropna()
    if table.empty:
        return 0, float("nan")
    best = table.loc[table["correlation"].abs().idxmax()]
    return int(best["lag"]), float(best["correlation"])
