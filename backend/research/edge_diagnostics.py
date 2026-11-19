"""Where does an intraday strategy's edge actually live? Measures every signal on its own (one open
position per strategy/symbol at a time, like shadow tracking), with live-parity fills (next bar's open),
gap-aware stops and Angel One's real costs at a fixed ticket - and records the conditions it fired under
(time of day, stop width, side, daily trend, regime, volatility, volume, target distance), under two exit
styles. The result is one row per signal, meant to be sliced: which conditions have a GROSS edge big
enough to survive costs, in both an earlier discovery period and a later holdout period.

Strategies are run with their own gating switched off (daily_trend_filter, allowed_regimes) so those
conditions can be measured rather than assumed.
"""
from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd

from backend.core.models import Side, Signal
from backend.execution import cost_model as cm
from backend.research.simple_backtester import entry_fill, scan_exit
from backend.strategies.base import Strategy, StrategyContext

UNGATED = {"daily_trend_filter": False, "allowed_regimes": None}
EXIT_VARIANTS = ("designed", "no_target")   # with the strategy's target / stop + square-off only


def signal_outcomes(feat: pd.DataFrame, strategy: Strategy, symbol: str, timeframe: str,
                    ticket: float = 200_000.0, square_off_time: time | None = time(15, 15)) -> list[dict]:
    signals = strategy.run(StrategyContext(symbol, timeframe, feat))
    pos = {t: i for i, t in enumerate(feat["ts"])}
    rows: list[dict] = []
    busy_until = None
    for sig in sorted(signals, key=lambda s: s.ts):
        if busy_until is not None and sig.ts <= busy_until:
            continue
        i = pos.get(sig.ts)
        if i is None:
            continue
        fill = entry_fill(feat, i, sig, "next_open")
        if fill is None or (sig.side == Side.BUY and sig.stop_loss >= fill) or \
                (sig.side == Side.SELL and sig.stop_loss <= fill):
            continue
        row = feat.iloc[i]
        qty = max(1, int(ticket // fill))
        stop_dist = abs(fill - sig.stop_loss)
        risk = qty * stop_dist
        atr = float(row["atr_14"]) if pd.notna(row.get("atr_14")) else 0.0
        target = sig.targets[0] if sig.targets else None
        rec = {
            "strategy": strategy.strategy_id, "symbol": symbol, "ts": sig.ts, "side": sig.side.value,
            "minute": (sig.ts.hour * 60 + sig.ts.minute) - (9 * 60 + 15),
            "stop_pct": stop_dist / fill * 100, "atr_pct": atr / fill * 100,
            "target_r": (abs(target - fill) / stop_dist) if target is not None else np.nan,
            "daily_trend": row.get("daily_trend", np.nan), "regime": row.get("regime"),
            "vol_ratio": (row["volume"] / row["avg_volume"]) if row.get("avg_volume") else np.nan,
            "with_trend": _with_trend(sig, row.get("daily_trend", np.nan)),
        }
        exit_designed = None
        for variant in EXIT_VARIANTS:
            tgt = target if variant == "designed" else None
            exit_ts, exit_px, reason = scan_exit(feat, i, sig.side, sig.stop_loss, tgt, sig.horizon, square_off_time)
            direction = 1 if sig.side == Side.BUY else -1
            gross = direction * (exit_px - fill) * qty
            cost = cm.round_trip_cost(fill, exit_px, qty, sig.side, sig.horizon).total + cm.slippage(fill, atr, qty)
            rec[f"gross_r_{variant}"] = gross / risk
            rec[f"net_r_{variant}"] = (gross - cost) / risk
            rec[f"cost_r_{variant}"] = cost / risk
            rec[f"net_rs_{variant}"] = gross - cost
            rec[f"exit_{variant}"] = reason
            if variant == "designed":
                exit_designed = exit_ts
        busy_until = exit_designed
        rows.append(rec)
    return rows


def _with_trend(sig: Signal, trend) -> float:
    if trend is None or pd.isna(trend):
        return np.nan
    return 1.0 if (trend > 0) == (sig.side == Side.BUY) else 0.0


def summarize(df: pd.DataFrame, by: list[str], variant: str = "designed") -> pd.DataFrame:
    g = df.groupby(by, observed=True)
    out = g.agg(n=(f"net_r_{variant}", "size"), gross_r=(f"gross_r_{variant}", "mean"),
                cost_r=(f"cost_r_{variant}", "mean"), net_r=(f"net_r_{variant}", "mean"),
                net_rs=(f"net_rs_{variant}", "sum"))
    return out.round(3)
