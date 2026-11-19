"""Market regime classification: trending / ranging / high-volatility state, used to decide which
strategies are allowed to trade (Phase 4 ensemble) and to score pattern context (context_score.py).

Two approaches, both lookahead-safe (every value at bar i uses only bars <= i):
  * `classify_regime` - a simple, explainable rule using ADX (trend strength) and an ATR percentile
    (volatility level), optionally overridden by India VIX. This is the primary, default classifier.
  * `statistical_regime` - an unsupervised Gaussian Mixture over rolling return mean/volatility,
    refit periodically (not every bar - a regime shouldn't flicker bar to bar). A secondary, data-driven
    cross-check rather than a replacement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRENDING_UP, TRENDING_DOWN, RANGING, HIGH_VOL = "trending_up", "trending_down", "ranging", "high_vol"


def atr_percentile(atr: pd.Series, lookback: int = 100) -> pd.Series:
    """Rolling percentile rank (0-1) of the current ATR within the trailing `lookback` bars (inclusive
    of the current bar, using only past+current data - no lookahead)."""
    def rank_last(window: np.ndarray) -> float:
        return (window <= window[-1]).mean()
    min_periods = min(lookback, max(10, lookback // 4))
    return atr.rolling(lookback, min_periods=min_periods).apply(rank_last, raw=True)


def classify_regime(adx: pd.Series, plus_di: pd.Series, minus_di: pd.Series, atr_pct: pd.Series,
                    trend_adx_min: float = 25.0, high_vol_pct: float = 0.85) -> pd.Series:
    """One label per bar. Volatility takes priority: an extreme-ATR bar is labelled `high_vol`
    regardless of trend strength (a strong trend during a volatility spike is still risky to trade
    normally). Otherwise: trending (up/down) if ADX clears the threshold, else ranging."""
    trending = adx >= trend_adx_min
    up = trending & (plus_di >= minus_di)
    down = trending & (plus_di < minus_di)
    label = pd.Series(RANGING, index=adx.index, dtype=object)
    label[up] = TRENDING_UP
    label[down] = TRENDING_DOWN
    label[atr_pct >= high_vol_pct] = HIGH_VOL
    label[adx.isna() | atr_pct.isna()] = None
    return label


def vix_regime(vix_close: pd.Series, high_percentile: float = 0.80, lookback: int = 252) -> pd.Series:
    """Market-wide risk state from India VIX: "risk_off" when VIX is in its own high percentile range
    over the trailing `lookback` days, else "normal". Meant to gate the whole portfolio (Phase 4 risk
    manager), separate from the per-stock regime above."""
    pct = atr_percentile(vix_close, lookback)   # same rolling-percentile math applies to any series
    return pd.Series(np.where(pct >= high_percentile, "risk_off", "normal"), index=vix_close.index)


# ---------------------------------------------------------------------- statistical (secondary) regime
def statistical_regime(close: pd.Series, window: int = 20, refit_every: int = 20, n_states: int = 3,
                       min_history: int = 250, random_state: int = 0) -> pd.Series:
    """Cluster rolling (mean return, volatility) pairs into `n_states` regimes with a Gaussian Mixture,
    refit only every `refit_every` bars using data available up to that point (no lookahead: a fit at
    bar i only sees returns up to bar i, and is applied going forward until the next refit).

    States are relabelled by volatility so results are comparable across refits: "low_vol", "mid_vol",
    "high_vol" for n_states=3 (for other `n_states`, labels are "vol_0".."vol_{n-1}", low to high).
    Returns NaN/None for bars before enough history has accumulated.
    """
    from sklearn.mixture import GaussianMixture

    ret = close.pct_change()
    mean_r = ret.rolling(window).mean()
    vol_r = ret.rolling(window).std(ddof=0)
    feat = pd.concat([mean_r, vol_r], axis=1).to_numpy()
    n = len(close)
    labels = np.array([None] * n, dtype=object)
    names = (["low_vol", "mid_vol", "high_vol"] if n_states == 3
            else [f"vol_{i}" for i in range(n_states)])

    last_fit_end = -1
    model = None
    order = None
    for i in range(min_history, n, refit_every):
        train = feat[:i]
        valid = ~np.isnan(train).any(axis=1)
        if valid.sum() < n_states * 5:
            continue
        model = GaussianMixture(n_components=n_states, random_state=random_state, n_init=2)
        model.fit(train[valid])
        # order cluster ids by their volatility-feature mean, low to high, so labels are comparable
        order = np.argsort(model.means_[:, 1])
        end = min(n, i + refit_every)
        window_feat = feat[i:end]
        window_valid = ~np.isnan(window_feat).any(axis=1)
        if window_valid.any():
            pred = model.predict(window_feat[window_valid])
            rank = {cid: r for r, cid in enumerate(order)}
            idxs = np.arange(i, end)[window_valid]
            for idx, cid in zip(idxs, pred):
                labels[idx] = names[rank[cid]]
        last_fit_end = end
    return pd.Series(labels, index=close.index, dtype=object)
