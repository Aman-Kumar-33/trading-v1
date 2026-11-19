"""Step 1 of the whipsaw study: collect every trade of a few base settings together with what the MACD / signal lines were
doing when it was entered (how often they had crossed lately, how close they had been), so we can see whether the losing
round trips really come from "the lines hugging each other for a long time".

    python scripts/tolerance_study_whipsaw.py            # writes research_data/tolerance_study/diag_trades.parquet and prints the tables
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.core.config import load_settings  # noqa: E402
from backend.data import trending  # noqa: E402
from backend.research import live_macd_replay as R  # noqa: E402
from tolerance_study import FLOOR, stock_data  # noqa: E402

BASES = [(0.02, 60), (0.0, 60), (0.1, 120)]
COST = 7e-4


def features_at(bs: R.BarState, window_list=(6, 12, 24)) -> dict:
    out = {}
    for w in window_list:
        out[f"chop{w}"] = R.chop_count(bs.gap, w)
        out[f"hug{w}"] = R.hug_ratio(bs.gap, bs.atr, w)
    sign = np.sign(np.where(np.isnan(bs.gap), 0.0, bs.gap))
    flips = np.r_[False, sign[1:] * sign[:-1] < 0]
    idx = np.where(flips, np.arange(len(sign)), -1)
    last_flip = np.maximum.accumulate(idx)
    out["since_cross"] = np.where(last_flip >= 0, np.arange(len(sign)) - last_flip, 999)
    return out


def collect(root: Path, symbols, log=print) -> pd.DataFrame:
    rows = []
    for n, sym in enumerate(symbols, 1):
        got = stock_data(root, sym)
        if got is None:
            continue
        df1, ok_days = got
        bs5 = R.bar_state(R.to_bars(df1, 5))
        tk = R.pseudo_ticks(df1, bs5, 5, FLOOR, "auto")
        feats = features_at(bs5)
        mod_of = (pd.to_datetime(tk.secs, unit="s", utc=True).tz_convert("Asia/Kolkata"))
        for k, h in BASES:
            for d, e, x, t_in, t_out, i_in, i_out in R.simulate(tk, k, h, FLOOR):
                day = tk.day_labels[d]
                if day not in ok_days:
                    continue
                j = int(tk.bar[i_in])
                rows.append(dict(symbol=sym, k=k, hold=h, date=day, ret=x / e - 1, dur_min=(t_out - t_in) / 60.0,
                                 entry_min=mod_of[i_in].hour * 60 + mod_of[i_in].minute,
                                 gap_atr=tk.gap[i_in] / max(tk.atr[i_in], 1e-9), **{key: v[j] for key, v in feats.items()}))
        log(f"  [{n}/{len(symbols)}] {sym}")
    return pd.DataFrame(rows)


def main() -> int:
    settings = load_settings()
    root = settings.paths.root / "research_data" / "angel_1m"
    out = settings.paths.root / "research_data" / "tolerance_study"
    df = collect(root, trending.universe_symbols(settings))
    df.to_parquet(out / "diag_trades.parquet")
    print(f"{len(df):,} trades saved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
