"""Chart patterns - built from the confirmed swing-point (zigzag) sequence, not from raw candles.

Every pattern here is a two-step process:
  1. formation - a sequence of recent confirmed swing points matches a geometric template (within an
     ATR-relative tolerance, so it works the same on a Rs 50 stock and a Rs 5,000 stock).
  2. confirmation - price actually breaks the level the pattern implies (neckline, trendline,
     range edge). The pattern is only reported - and only becomes a signal - at the breakout bar.
     A "formed but not yet broken out" pattern is not a signal.

This mirrors real trading: a head-and-shoulders that never breaks its neckline isn't a trade.
Everything only uses swing points already confirmed by `backend.features.price_action.swings` +
`confirmed_at`, and only looks at bars up to the breakout bar - no lookahead.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

BULLISH, BEARISH, NONE = 100, -100, 0


@dataclass
class Pivot:
    idx: int          # positional index (row number) in the OHLC frame
    price: float
    kind: str          # "H" or "L"


def zigzag_pivots(swing_high_confirmed: pd.Series, swing_low_confirmed: pd.Series,
                  swing_high_price: pd.Series, swing_low_price: pd.Series) -> list[Pivot]:
    """Merge confirmed swing highs/lows into a single alternating H/L/H/L... sequence.

    Fractal detection can mark two highs (or two lows) in a row with no opposite pivot between them
    (e.g. a higher high forms before the prior high's "opposite" low is confirmed). When that happens,
    keep only the more extreme of the two - a zigzag can't have two consecutive same-type turns.
    """
    hi = swing_high_confirmed.to_numpy(bool)
    lo = swing_low_confirmed.to_numpy(bool)
    hp = swing_high_price.to_numpy(float)
    lp = swing_low_price.to_numpy(float)
    events: list[Pivot] = []
    for i in range(len(hi)):
        if hi[i] and not np.isnan(hp[i]):
            events.append(Pivot(i, hp[i], "H"))
        if lo[i] and not np.isnan(lp[i]):
            events.append(Pivot(i, lp[i], "L"))
    events.sort(key=lambda p: p.idx)

    pivots: list[Pivot] = []
    for p in events:
        if pivots and pivots[-1].kind == p.kind:
            if (p.kind == "H" and p.price >= pivots[-1].price) or (p.kind == "L" and p.price <= pivots[-1].price):
                pivots[-1] = p       # more extreme - replaces the previous same-type pivot
            # else: less extreme than the standing pivot - ignored
        else:
            pivots.append(p)
    return pivots


def _close_enough(a: float, b: float, atr: float, tol_atr: float) -> bool:
    return abs(a - b) <= tol_atr * atr


def _atr_at(atr: pd.Series, idx: int) -> float:
    v = atr.iloc[idx]
    return float(v) if not np.isnan(v) else float(atr.dropna().iloc[0]) if atr.notna().any() else 1.0


def _first_breakout(close: pd.Series, from_idx: int, level: float,
                    direction: Literal["above", "below"], max_bars: int) -> int | None:
    """First bar after `from_idx` (exclusive) whose *close* breaks the level, searched up to
    `max_bars` ahead. Returns the breakout bar's index, or None if it never happens in time."""
    end = min(len(close), from_idx + 1 + max_bars)
    for i in range(from_idx + 1, end):
        if direction == "above" and close.iloc[i] > level:
            return i
        if direction == "below" and close.iloc[i] < level:
            return i
    return None


def _empty_signal(index: pd.Index) -> pd.Series:
    return pd.Series(NONE, index=index)


# ============================================================================ double / triple tops & bottoms
def double_top_bottom(pivots: list[Pivot], close: pd.Series, atr: pd.Series, index: pd.Index,
                      tol_atr: float = 0.5, min_separation: int = 3, max_breakout_bars: int = 20) -> pd.Series:
    """Double Top: H1, L (valley), H2 with H2~=H1 -> bearish break below the valley.
    Double Bottom: mirror (bullish break above the peak between the two lows)."""
    out = _empty_signal(index)
    for i in range(2, len(pivots)):
        p2, mid, p1 = pivots[i], pivots[i - 1], pivots[i - 2]
        if p2.kind != p1.kind or mid.kind == p1.kind or p2.idx - p1.idx < min_separation:
            continue
        a = _atr_at(atr, p2.idx)
        if not _close_enough(p1.price, p2.price, a, tol_atr):
            continue
        if p1.kind == "H":   # double top
            brk = _first_breakout(close, p2.idx, mid.price, "below", max_breakout_bars)
            if brk is not None:
                out.iloc[brk] = BEARISH
        else:                # double bottom
            brk = _first_breakout(close, p2.idx, mid.price, "above", max_breakout_bars)
            if brk is not None:
                out.iloc[brk] = BULLISH
    return out


def triple_top_bottom(pivots: list[Pivot], close: pd.Series, atr: pd.Series, index: pd.Index,
                      tol_atr: float = 0.5, min_separation: int = 3, max_breakout_bars: int = 20) -> pd.Series:
    """Triple Top: H1~H2~H3 with two similar valleys between them -> break below the lower valley.
    Triple Bottom: mirror."""
    out = _empty_signal(index)
    for i in range(4, len(pivots)):
        p5, v2, p4, v1, p3 = pivots[i], pivots[i - 1], pivots[i - 2], pivots[i - 3], pivots[i - 4]
        if not (p5.kind == p4.kind == p3.kind and v1.kind == v2.kind and v1.kind != p3.kind):
            continue
        if p5.idx - p3.idx < min_separation:
            continue
        a = _atr_at(atr, p5.idx)
        if not (_close_enough(p3.price, p4.price, a, tol_atr) and _close_enough(p4.price, p5.price, a, tol_atr)
               and _close_enough(v1.price, v2.price, a, tol_atr)):
            continue
        if p3.kind == "H":
            level = min(v1.price, v2.price)
            brk = _first_breakout(close, p5.idx, level, "below", max_breakout_bars)
            if brk is not None:
                out.iloc[brk] = BEARISH
        else:
            level = max(v1.price, v2.price)
            brk = _first_breakout(close, p5.idx, level, "above", max_breakout_bars)
            if brk is not None:
                out.iloc[brk] = BULLISH
    return out


# ============================================================================ head & shoulders
def head_shoulders(pivots: list[Pivot], close: pd.Series, atr: pd.Series, index: pd.Index,
                   shoulder_tol_atr: float = 0.6, min_head_atr: float = 0.5,
                   max_breakout_bars: int = 20) -> pd.Series:
    """H&S: left shoulder, trough, head (higher than both shoulders), trough, right shoulder (~= left
    shoulder). Neckline = line through the two troughs. Bearish break below the neckline (projected to
    the breakout bar's position). Inverse H&S is the exact mirror (bullish break above the neckline)."""
    out = _empty_signal(index)
    for i in range(4, len(pivots)):
        rs, t2, head, t1, ls = pivots[i], pivots[i - 1], pivots[i - 2], pivots[i - 3], pivots[i - 4]
        if not (rs.kind == ls.kind == head.kind and t1.kind == t2.kind and t1.kind != head.kind):
            continue
        a = _atr_at(atr, rs.idx)
        if not _close_enough(ls.price, rs.price, a, shoulder_tol_atr):
            continue
        if head.kind == "H":
            if not (head.price > ls.price + min_head_atr * a and head.price > rs.price + min_head_atr * a):
                continue
            neckline = _project_line(t1, t2, rs.idx + 1)
            brk = _first_breakout(close, rs.idx, neckline, "below", max_breakout_bars)
            if brk is not None:
                out.iloc[brk] = BEARISH
        else:
            if not (head.price < ls.price - min_head_atr * a and head.price < rs.price - min_head_atr * a):
                continue
            neckline = _project_line(t1, t2, rs.idx + 1)
            brk = _first_breakout(close, rs.idx, neckline, "above", max_breakout_bars)
            if brk is not None:
                out.iloc[brk] = BULLISH
    return out


def _project_line(p1: Pivot, p2: Pivot, at_idx: int) -> float:
    """Value of the straight line through two pivots, projected to bar `at_idx` (handles a sloped
    neckline; degenerates to a flat line when the two points share a bar index)."""
    if p2.idx == p1.idx:
        return (p1.price + p2.price) / 2
    slope = (p2.price - p1.price) / (p2.idx - p1.idx)
    return p2.price + slope * (at_idx - p2.idx)


# ============================================================================ rectangles (ranges)
def rectangle(pivots: list[Pivot], close: pd.Series, atr: pd.Series, index: pd.Index,
             tol_atr: float = 0.4, min_touches_each_side: int = 2, max_breakout_bars: int = 20) -> pd.Series:
    """At least `min_touches_each_side` highs cluster near one level and the same number of lows
    cluster near another -> a trading range. Breakout above the resistance is bullish, below support
    is bearish. Only the most recent qualifying window ending at each pivot is checked (cheap, and
    matches how a trader would actually notice the range as it forms)."""
    out = _empty_signal(index)
    window = min_touches_each_side * 2
    for i in range(window - 1, len(pivots)):
        recent = pivots[i - window + 1: i + 1]
        highs = [p.price for p in recent if p.kind == "H"]
        lows = [p.price for p in recent if p.kind == "L"]
        if len(highs) < min_touches_each_side or len(lows) < min_touches_each_side:
            continue
        a = _atr_at(atr, recent[-1].idx)
        if (max(highs) - min(highs)) > tol_atr * a or (max(lows) - min(lows)) > tol_atr * a:
            continue
        resistance, support = float(np.mean(highs)), float(np.mean(lows))
        if resistance - support < 2 * tol_atr * a:      # too thin to be a meaningful range
            continue
        start = recent[-1].idx
        up = _first_breakout(close, start, resistance, "above", max_breakout_bars)
        down = _first_breakout(close, start, support, "below", max_breakout_bars)
        if up is not None and (down is None or up <= down):
            out.iloc[up] = BULLISH
        elif down is not None:
            out.iloc[down] = BEARISH
    return out


# ============================================================================ triangles / wedges / channels
@dataclass
class TrendLine:
    slope: float
    intercept: float   # value at idx=0

    def at(self, idx: int) -> float:
        return self.slope * idx + self.intercept


def fit_trendline(pivots: list[Pivot]) -> TrendLine | None:
    if len(pivots) < 2:
        return None
    x = np.array([p.idx for p in pivots], dtype=float)
    y = np.array([p.price for p in pivots], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return TrendLine(float(slope), float(intercept))


def _classify_two_lines(upper: TrendLine, lower: TrendLine, atr_val: float, flat_slope_atr: float
                        ) -> Literal["ascending", "descending", "symmetrical", "rising_wedge",
                                    "falling_wedge", "channel", None]:
    u, l = upper.slope, lower.slope
    flat = flat_slope_atr * atr_val
    if abs(u) <= flat and l > flat:
        return "ascending"
    if u < -flat and abs(l) <= flat:
        return "descending"
    if u < -flat and l > flat:
        return "symmetrical"
    if u > flat and l > flat and u < l:
        return "rising_wedge"        # both rising, converging (upper rises slower than lower catches up)
    if u < -flat and l < -flat and u < l:
        return "falling_wedge"       # both falling, converging (upper falls faster, catching down to lower)
    if abs(u - l) <= flat:
        return "channel"             # parallel lines, trending together
    return None


def triangles_wedges_channels(pivots: list[Pivot], close: pd.Series, atr: pd.Series, index: pd.Index,
                              n_pivots: int = 4, flat_slope_atr: float = 0.05,
                              max_breakout_bars: int = 20) -> pd.DataFrame:
    """Fit a trendline through the recent swing highs and another through the recent swing lows,
    classify the shape from their slopes, and report the breakout once price closes beyond whichever
    line it's converging toward (or, for a channel/rising/falling wedge, beyond the near line).

    Returns a DataFrame with a `pattern` column (the shape name) and a `signal` column (+100/-100/0),
    both indexed like the input - most bars have pattern=None/signal=0.
    """
    pattern_col = pd.Series([None] * len(index), index=index, dtype=object)
    signal_col = _empty_signal(index)
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
        if shape in ("ascending", "channel") or (shape == "falling_wedge"):
            brk = _first_breakout_line(close, start, upper, "above", max_breakout_bars)
            if brk is not None:
                pattern_col.iloc[brk] = shape
                signal_col.iloc[brk] = BULLISH
                continue
        if shape in ("descending", "channel", "rising_wedge"):
            brk = _first_breakout_line(close, start, lower, "below", max_breakout_bars)
            if brk is not None:
                pattern_col.iloc[brk] = shape
                signal_col.iloc[brk] = BEARISH
                continue
        if shape == "symmetrical":
            brk_up = _first_breakout_line(close, start, upper, "above", max_breakout_bars)
            brk_dn = _first_breakout_line(close, start, lower, "below", max_breakout_bars)
            if brk_up is not None and (brk_dn is None or brk_up <= brk_dn):
                pattern_col.iloc[brk_up] = shape
                signal_col.iloc[brk_up] = BULLISH
            elif brk_dn is not None:
                pattern_col.iloc[brk_dn] = shape
                signal_col.iloc[brk_dn] = BEARISH
    return pd.DataFrame({"pattern": pattern_col, "signal": signal_col})


def _first_breakout_line(close: pd.Series, from_idx: int, line: TrendLine, direction: str, max_bars: int) -> int | None:
    end = min(len(close), from_idx + 1 + max_bars)
    for i in range(from_idx + 1, end):
        level = line.at(i)
        if direction == "above" and close.iloc[i] > level:
            return i
        if direction == "below" and close.iloc[i] < level:
            return i
    return None


# ============================================================================ flags / pennants
def flag_pennant(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, atr: pd.Series,
                 pole_bars: int = 5, pole_atr_min: float = 3.0, flag_bars: int = 8,
                 flag_max_range_atr: float = 2.5, max_breakout_bars: int = 10) -> pd.Series:
    """Continuation pattern: a sharp "pole" move, then a short, low-range consolidation ("flag" - drifts
    gently against the pole; "pennant" - contracts symmetrically; both handled the same way here: a
    tight-range pause), then a breakout continuing the pole's direction."""
    out = _empty_signal(close.index)
    n = len(close)
    for end_pole in range(pole_bars, n - flag_bars - 1):
        start_pole = end_pole - pole_bars
        a = _atr_at(atr, end_pole)
        if a <= 0 or np.isnan(a):
            continue
        pole_move = close.iloc[end_pole] - close.iloc[start_pole]
        if abs(pole_move) < pole_atr_min * a:
            continue
        flag_start, flag_end = end_pole, end_pole + flag_bars
        flag_high = high.iloc[flag_start:flag_end].max()
        flag_low = low.iloc[flag_start:flag_end].min()
        if (flag_high - flag_low) > flag_max_range_atr * a:
            continue   # consolidation too wide - not a tight flag/pennant
        # the flag must not fully retrace the pole
        if pole_move > 0 and flag_low < close.iloc[start_pole]:
            continue
        if pole_move < 0 and flag_high > close.iloc[start_pole]:
            continue
        direction = "above" if pole_move > 0 else "below"
        level = flag_high if direction == "above" else flag_low
        brk = _first_breakout(close, flag_end - 1, level, direction, max_breakout_bars)
        if brk is not None:
            out.iloc[brk] = BULLISH if pole_move > 0 else BEARISH
    return out


# ============================================================================ cup & handle
def cup_handle(close: pd.Series, high: pd.Series, low: pd.Series, atr: pd.Series,
              cup_min_bars: int = 20, cup_max_bars: int = 120, rim_tol_atr: float = 0.6,
              min_depth_atr: float = 2.0, handle_max_bars: int = 15, handle_max_retrace: float = 0.5,
              max_breakout_bars: int = 15) -> pd.Series:
    """Rounded U-shaped recovery (left rim -> cup bottom -> right rim near the left rim's level),
    followed by a shallow "handle" pullback, then a breakout above the rim. A simplified but usable
    version of a famously fuzzy pattern - the cup's roundedness is approximated by requiring the bottom
    to be a genuine local minimum roughly centred in the window, not a V-shaped spike."""
    out = _empty_signal(close.index)
    n = len(close)
    for right_rim in range(cup_min_bars, n - handle_max_bars - 1):
        a = _atr_at(atr, right_rim)
        found = False
        for cup_len in range(cup_min_bars, min(cup_max_bars, right_rim) + 1):
            left_rim = right_rim - cup_len
            segment_low_idx = low.iloc[left_rim:right_rim + 1].idxmin()
            bottom_pos = close.index.get_loc(segment_low_idx) if not isinstance(close.index, pd.RangeIndex) else segment_low_idx
            # bottom should sit roughly in the middle third of the cup (rounded, not a V at one edge)
            rel = (bottom_pos - left_rim) / cup_len
            if not (0.25 <= rel <= 0.75):
                continue
            left_price, right_price, bottom_price = close.iloc[left_rim], close.iloc[right_rim], low.iloc[bottom_pos]
            if not _close_enough(left_price, right_price, a, rim_tol_atr):
                continue
            depth = min(left_price, right_price) - bottom_price
            if depth < min_depth_atr * a:
                continue
            found = True
            break
        if not found:
            continue
        rim_level = max(close.iloc[left_rim], close.iloc[right_rim])
        handle_end = min(n - 1, right_rim + handle_max_bars)
        handle_low = low.iloc[right_rim:handle_end + 1].min()
        if rim_level - handle_low > handle_max_retrace * depth:
            continue    # handle pulled back too far - invalidates the setup
        brk = _first_breakout(close, handle_end - 1, rim_level, "above", max_breakout_bars)
        if brk is not None:
            out.iloc[brk] = BULLISH
    return out
