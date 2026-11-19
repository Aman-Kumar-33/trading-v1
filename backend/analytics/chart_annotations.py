"""Classic technical-analysis "measuring objective" entry/stop/target annotations for the dashboard's
live Charts page (source: chart tricks/Technical-analysis-Price-patterns.pdf - Head & Shoulders, Double
Top/Bottom, Triangles).

This is a DISPLAY-ONLY overlay, deliberately kept separate from `backend.features.patterns.chart_patterns`
(which strategies and the backtester use) - it re-derives the same pivot geometry with its own small
breakout loop rather than importing the boolean signal functions, so this purely visual annotation layer
can never accidentally change what a live strategy actually trades. Real order placement keeps using the
already cost-validated R-multiple stop/target logic in `backend.strategies.base` (see Phase 4.8's intraday
edge audit) - turning these classical levels into real orders would need its own backtest validation,
which is a separate decision, not something a chart overlay should silently do.

Rules straight from the PDF:
  - Head & Shoulders: entry = close beyond the neckline; stop = beyond the head ("head taken off");
    target = neckline-to-head height, projected from the breakout.
  - Double Top/Bottom: entry = close beyond the valley/peak between the two tops/bottoms; stop = beyond
    the second top/bottom; target = pattern height, projected from the breakout.
  - Triangles/wedges/channels: entry = close beyond the converging trendline; stop = the opposite
    trendline's value at the breakout bar; target = the pattern's vertical height, projected from the
    breakout.
"""
from __future__ import annotations

import pandas as pd

from backend.features.patterns.chart_patterns import (
    _atr_at, _classify_two_lines, _close_enough, _first_breakout, _first_breakout_line, _project_line,
    fit_trendline, zigzag_pivots,
)

MAX_ANNOTATIONS = 3   # most recent patterns only - a chart with every historical H&S ever found is noise

REQUIRED_COLUMNS = {"swing_high_confirmed", "swing_low_confirmed", "swing_high_price_confirmed",
                    "swing_low_price_confirmed", "atr_14", "close"}


def _head_shoulders(pivots, close, atr, shoulder_tol_atr=0.6, min_head_atr=0.5, max_breakout_bars=20):
    out = []
    for i in range(4, len(pivots)):
        rs, t2, head, t1, ls = pivots[i], pivots[i - 1], pivots[i - 2], pivots[i - 3], pivots[i - 4]
        if not (rs.kind == ls.kind == head.kind and t1.kind == t2.kind and t1.kind != head.kind):
            continue
        a = _atr_at(atr, rs.idx)
        if not _close_enough(ls.price, rs.price, a, shoulder_tol_atr):
            continue
        bearish = head.kind == "H"
        if bearish and not (head.price > ls.price + min_head_atr * a and head.price > rs.price + min_head_atr * a):
            continue
        if not bearish and not (head.price < ls.price - min_head_atr * a and head.price < rs.price - min_head_atr * a):
            continue
        neckline = _project_line(t1, t2, rs.idx + 1)
        brk = _first_breakout(close, rs.idx, neckline, "below" if bearish else "above", max_breakout_bars)
        if brk is None:
            continue
        entry = float(close.iloc[brk])
        height = abs(head.price - _project_line(t1, t2, head.idx))
        out.append({"type": "head_shoulders" if bearish else "inverse_head_shoulders",
                   "side": "bearish" if bearish else "bullish", "breakout_idx": brk,
                   "neckline_points": [[t1.idx, t1.price], [t2.idx, t2.price]], "entry": entry,
                   "stop": head.price, "target": entry - height if bearish else entry + height})
    return out


def _double_top_bottom(pivots, close, atr, tol_atr=0.5, min_separation=3, max_breakout_bars=20):
    out = []
    for i in range(2, len(pivots)):
        p2, mid, p1 = pivots[i], pivots[i - 1], pivots[i - 2]
        if p2.kind != p1.kind or mid.kind == p1.kind or p2.idx - p1.idx < min_separation:
            continue
        a = _atr_at(atr, p2.idx)
        if not _close_enough(p1.price, p2.price, a, tol_atr):
            continue
        bearish = p1.kind == "H"
        brk = _first_breakout(close, p2.idx, mid.price, "below" if bearish else "above", max_breakout_bars)
        if brk is None:
            continue
        entry = float(close.iloc[brk])
        height = abs(p2.price - mid.price)
        out.append({"type": "double_top" if bearish else "double_bottom", "side": "bearish" if bearish else "bullish",
                   "breakout_idx": brk, "neckline_points": [[mid.idx, mid.price], [brk, mid.price]],
                   "entry": entry, "stop": p2.price, "target": entry - height if bearish else entry + height})
    return out


def _triangles(pivots, close, atr, n_pivots=4, flat_slope_atr=0.05, max_breakout_bars=20):
    out = []
    for i in range(n_pivots - 1, len(pivots)):
        window = pivots[i - n_pivots + 1: i + 1]
        highs = [p for p in window if p.kind == "H"]
        lows = [p for p in window if p.kind == "L"]
        if len(highs) < 2 or len(lows) < 2:
            continue
        upper, lower = fit_trendline(highs), fit_trendline(lows)
        if upper is None or lower is None:
            continue
        a = _atr_at(atr, window[-1].idx)
        shape = _classify_two_lines(upper, lower, a, flat_slope_atr)
        if shape is None:
            continue
        start = window[-1].idx
        height = abs(upper.at(start) - lower.at(start))

        def emit(brk, line, bullish):
            entry = float(close.iloc[brk])
            out.append({"type": shape, "side": "bullish" if bullish else "bearish", "breakout_idx": brk,
                       "trendline_points": [[window[0].idx, line.at(window[0].idx)], [brk, line.at(brk)]],
                       "entry": entry, "stop": float((lower if bullish else upper).at(brk)),
                       "target": entry + height if bullish else entry - height})

        if shape in ("ascending", "channel", "falling_wedge"):
            brk = _first_breakout_line(close, start, upper, "above", max_breakout_bars)
            if brk is not None:
                emit(brk, upper, True)
                continue
        if shape in ("descending", "channel", "rising_wedge"):
            brk = _first_breakout_line(close, start, lower, "below", max_breakout_bars)
            if brk is not None:
                emit(brk, lower, False)
                continue
        if shape == "symmetrical":
            brk_up = _first_breakout_line(close, start, upper, "above", max_breakout_bars)
            brk_dn = _first_breakout_line(close, start, lower, "below", max_breakout_bars)
            if brk_up is not None and (brk_dn is None or brk_up <= brk_dn):
                emit(brk_up, upper, True)
            elif brk_dn is not None:
                emit(brk_dn, lower, False)
    return out


def chart_annotations(feat: pd.DataFrame) -> list[dict]:
    """`feat` must be `backend.features.pipeline.build_features(df, timeframe)` output (needs the swing
    columns + atr_14 it computes - core_only=True frames won't have them). Returns the most recent
    `MAX_ANNOTATIONS` pattern entry/stop/target annotations, timestamped, for a chart to draw."""
    if feat is None or feat.empty or not REQUIRED_COLUMNS.issubset(feat.columns):
        return []
    pivots = zigzag_pivots(feat["swing_high_confirmed"], feat["swing_low_confirmed"],
                           feat["swing_high_price_confirmed"], feat["swing_low_price_confirmed"])
    close, atr = feat["close"], feat["atr_14"]
    events = _head_shoulders(pivots, close, atr) + _double_top_bottom(pivots, close, atr) + _triangles(pivots, close, atr)
    events.sort(key=lambda e: e["breakout_idx"])
    recent = events[-MAX_ANNOTATIONS:]
    ts = feat["ts"] if "ts" in feat.columns else pd.Series(feat.index, index=feat.index)
    for e in recent:
        e["breakout_ts"] = str(ts.iloc[e["breakout_idx"]])
        for key in ("neckline_points", "trendline_points"):
            if key in e:
                e[key] = [[str(ts.iloc[idx]), round(float(price), 4)] for idx, price in e[key]]
    return recent
