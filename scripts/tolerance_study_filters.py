"""Step 2 of the whipsaw study: ways of ignoring whipsaw trades, tried on top of the live rule, over every stock and day.

Each candidate is a setting of `Rule` (live_macd_replay): a looser/stricter EXIT tolerance, a gate that blocks a BUY while the
lines have been hugging / crossing a lot, a gate that needs the 15m MACD to agree, a cool-down after an exit, or a cap on
entries per day. Every trade is saved with its date, so the analysis can score the candidates on one period and check them
on a later one.

    python scripts/tolerance_study_filters.py       # -> research_data/tolerance_study/filter_trades.parquet
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.core.config import load_settings  # noqa: E402
from backend.data import trending  # noqa: E402
from backend.research import live_macd_replay as R  # noqa: E402
from tolerance_study import FLOOR, stock_data  # noqa: E402

BASES = [(0.02, 60), (0.1, 60)]           # (tolerance x ATR, hold seconds) the extras are added to
GATES = [None, "chop2", "chop3", "hug0.07", "hug0.10", "hug0.15", "htf15", "htf15_zero", "macd5_zero", "htf15+macd5_zero"]
EXIT_KS = [None, 0.0, 0.05, 0.2]          # None = same as the entry tolerance
COOLDOWNS = [0, 900, 1800, 3600]          # seconds
MAX_ENTRIES = [None, 1, 2]


def candidates():
    """(name, k, hold, rule kwargs) - one factor at a time, then a few combinations."""
    out = []
    for k, h in BASES:
        out.append((f"base", k, h, {}))
        for ek in EXIT_KS[1:]:
            out.append((f"exit_k={ek}", k, h, dict(exit_k=ek)))
        for g in GATES[1:]:
            out.append((f"gate={g}", k, h, dict(gate=g)))
        for c in COOLDOWNS[1:]:
            out.append((f"cooldown={c // 60}m", k, h, dict(cooldown=c)))
        for m in MAX_ENTRIES[1:]:
            out.append((f"max_entries={m}", k, h, dict(max_entries=m)))
        out += [(f"gate=htf15+exit_k=0.05", k, h, dict(gate="htf15", exit_k=0.05)),
                (f"gate=htf15+cooldown=30m", k, h, dict(gate="htf15", cooldown=1800)),
                (f"gate=chop3+cooldown=30m", k, h, dict(gate="chop3", cooldown=1800)),
                (f"gate=htf15+macd5_zero+cooldown=30m", k, h, dict(gate="htf15+macd5_zero", cooldown=1800))]
    return out


def bar_gates(df1: pd.DataFrame, bs5: R.BarState, tk: R.Ticks) -> dict:
    """Boolean per tick for every gate name (True = a BUY is allowed). Only information from CLOSED bars is used."""
    bars5 = bs5.bars
    bs15 = R.bar_state(R.to_bars(df1, 15))
    b15 = bs15.bars
    k5 = pd.to_datetime(bars5["day"]).map(pd.Timestamp.toordinal).to_numpy() * 10000 + bars5["start_min"].to_numpy()
    k15 = pd.to_datetime(b15["day"]).map(pd.Timestamp.toordinal).to_numpy() * 10000 + b15["end_min"].to_numpy()
    # the last 15m bar that had CLOSED when each 5m bar began
    pos = np.searchsorted(k15, k5, side="right") - 1
    ok15 = pos >= 0
    pos = np.where(ok15, pos, 0)
    gap15 = np.where(ok15, bs15.gap[pos], np.nan)
    macd15 = np.where(ok15, (bs15.e12 - bs15.e26)[pos], np.nan)
    fb = tk.bar + 1                                       # the forming 5m bar of each tick
    chop12, hug12 = R.chop_count(bs5.gap, 12), R.hug_ratio(bs5.gap, bs5.atr, 12)
    macd5 = bs5.e12 - bs5.e26
    c = tk.bar
    g = {
        "chop2": chop12[c] < 2, "chop3": chop12[c] < 3,
        "hug0.07": hug12[c] >= 0.07, "hug0.10": hug12[c] >= 0.10, "hug0.15": hug12[c] >= 0.15,
        "htf15": gap15[fb] > 0, "htf15_zero": macd15[fb] > 0, "macd5_zero": macd5[c] > 0,
    }
    g["htf15+macd5_zero"] = g["htf15"] & g["macd5_zero"]
    return {k: np.nan_to_num(v.astype(float)).astype(bool) for k, v in g.items()}


def main() -> int:
    settings = load_settings()
    root = settings.paths.root / "research_data" / "angel_1m"
    out = settings.paths.root / "research_data" / "tolerance_study"
    symbols = trending.universe_symbols(settings)
    cands = candidates()
    frames = []
    for n, sym in enumerate(symbols, 1):
        t0 = time.time()
        got = stock_data(root, sym)
        if got is None:
            continue
        df1, ok_days = got
        bs5 = R.bar_state(R.to_bars(df1, 5))
        tk = R.pseudo_ticks(df1, bs5, 5, FLOOR, "auto")
        gates = bar_gates(df1, bs5, tk)
        recs = []
        for name, k, h, kw in cands:
            kw = dict(kw)
            if "gate" in kw:
                kw["gate"] = gates[kw["gate"]]
            for d, e, x, t_in, t_out, i_in, i_out in R.simulate(tk, k, h, FLOOR, R.Rule(**kw)):
                day = tk.day_labels[d]
                if day in ok_days:
                    recs.append((sym, name, k, h, day, x / e - 1))
        frames.append(pd.DataFrame(recs, columns=["symbol", "cand", "k", "hold", "date", "ret"]).astype(
            {"symbol": "category", "cand": "category", "k": "float32", "hold": "int16", "ret": "float32"}))
        print(f"  [{n}/{len(symbols)}] {sym}: {len(recs)} trades, {time.time() - t0:.0f}s", flush=True)
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(out / "filter_trades.parquet")
    print(f"{len(df):,} trades, {len(cands)} candidates saved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
