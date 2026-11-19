"""MACD-only crossover, long/flat: BUY on the MACD line crossing ABOVE its signal line, SELL (exit) on it
crossing BELOW. A cross is detected on a closed bar `i` exactly as it reads on a chart - at bar i-1 the
MACD line was at/below the signal line and at bar i it is above (and the mirror for a bearish cross) - so
the signal is stamped with that bar's start time and can be checked by eye against any 30m chart.

No stop-loss/target layer - this sandbox system has no risk manager; the stop_loss field is set far away
purely to satisfy Signal's schema and is not meant to trigger before the next opposite signal does.

`entry_filters` (off by default) restores the older quality gate on BUYs only - histogram expanding plus
above-average volume. It is off because it silently skips real crosses, which makes the signals impossible
to verify against a chart one-for-one.
"""
from __future__ import annotations

import numpy as np

from backend.core.models import Horizon, Side, Signal
from backend.strategies.base import Strategy, StrategyContext


def macd_cross_masks(macd: np.ndarray, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(bullish, bearish) boolean arrays: True on the bar where the cross completes. NaN on either line
    (indicator still warming up) never produces a cross."""
    prev_macd, prev_sig = np.roll(macd, 1), np.roll(signal, 1)
    valid = ~(np.isnan(macd) | np.isnan(signal) | np.isnan(prev_macd) | np.isnan(prev_sig))
    valid[0] = False   # np.roll wraps the last bar around to index 0 - there is no previous bar there
    bull = valid & (prev_macd <= prev_sig) & (macd > signal)
    bear = valid & (prev_macd >= prev_sig) & (macd < signal)
    return bull, bear


def confirmed_cross_up(gap: np.ndarray, tol: float) -> bool:
    """True when the newest closed bar is the FIRST one with MACD more than `tol` above the signal line since the two lines
    were last at/below each other - i.e. the lines have intersected (MACD from below to above) and the cross has now cleared
    the tolerance. Bars in between that are above the line but within `tol` are the cross still waiting for confirmation;
    a bar already beyond `tol` earlier means this is the same up-move, not a fresh cross."""
    n = len(gap)
    if n < 2 or not gap[-1] > tol:
        return False
    for j in range(n - 2, -1, -1):
        g = gap[j]
        if np.isnan(g) or g > tol:
            return False
        if g <= 0:
            return True
    return False


class MacdOnly(Strategy):
    strategy_id = "macd_only"
    horizon = Horizon.INTRADAY
    needs_full_features = False
    default_params = {"wide_stop_atr_mult": 10.0, "entry_filters": False}

    def generate_signals(self, ctx: StrategyContext) -> list[Signal]:
        df = ctx.df
        p = self.params
        macd, sig = df["macd"].to_numpy(dtype=float), df["macd_signal"].to_numpy(dtype=float)
        bull, bear = macd_cross_masks(macd, sig)
        hist = df["macd_hist"].to_numpy(dtype=float)
        vol = df["volume"].to_numpy(dtype=float)
        avg_vol = df["avg_volume"].to_numpy(dtype=float) if "avg_volume" in df.columns else np.full(len(df), np.nan)
        out: list[Signal] = []
        for i in np.flatnonzero(bull | bear):
            is_buy = bool(bull[i])
            if is_buy and p["entry_filters"]:
                if not abs(hist[i]) > abs(hist[i - 1]) or (avg_vol[i] == avg_vol[i] and vol[i] <= avg_vol[i]):
                    continue
            row = df.iloc[i]
            entry = float(row["close"])
            atr = row.get("atr_14")
            atr = entry * 0.02 if atr is None or atr != atr else float(atr)
            stop = entry - p["wide_stop_atr_mult"] * atr if is_buy else entry + p["wide_stop_atr_mult"] * atr
            out.append(Signal(symbol=ctx.symbol, side=Side.BUY if is_buy else Side.SELL, horizon=self.horizon,
                              stop_loss=stop, strategy_id="", strategy_version="", ts=row["ts"],
                              timeframe=ctx.timeframe, entry=entry,
                              reason=f"MACD {'bullish' if is_buy else 'bearish'} cross "
                                     f"(macd {macd[i]:.4f} {'>' if is_buy else '<'} signal {sig[i]:.4f})"))
        return out
