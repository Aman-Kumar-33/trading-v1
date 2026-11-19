"""Does the trending picker (history-only, as picked before the open) choose stocks where the MACD rules do better?
Each day D: rank the NIFTY 50 on their last 10 sessions of 15m bars up to D-1, keep the top 10, and score the study's trades
of day D only for those stocks, against all 50 and against random 10-stock draws."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\Business\algo trading\v17 anthropic\second-system")
sys.path.insert(0, str(ROOT))
from backend.data import trending
from backend.research import live_macd_replay as R

COST = 7e-4
tr = pd.read_parquet(ROOT / "research_data/tolerance_study/trades.parquet")
tr["date"] = pd.to_datetime(tr["date"])
tr["symbol"] = tr["symbol"].astype(str)
days = np.array(sorted(tr["date"].unique()))
symbols = sorted(tr["symbol"].unique())

hist = {}
for s in symbols:
    df1 = R.load_1m(ROOT / "research_data/angel_1m" / s.replace("&", "_AND_"))
    b = R.to_bars(df1, 15)
    b = b.rename(columns={"ts": "ts"})
    b["ts"] = pd.to_datetime(b["ts"])
    b["volume"] = 0.0
    hist[s] = b[["ts", "open", "high", "low", "close", "volume", "day"]]

picks = {}
for i, d in enumerate(days):
    if i < 11:
        continue
    d0 = pd.Timestamp(d).date()
    h = {s: b[b["day"] < d0].drop(columns="day").reset_index(drop=True) for s, b in hist.items()}
    h = {s: v for s, v in h.items() if len(v) > 80}
    p = trending.rank_picks(h, {}, 10, max_price=250000.0)
    picks[pd.Timestamp(d)] = [x["symbol"] for x in p]
print(f"picked lists for {len(picks)} days; example {list(picks.items())[-1]}")

rng = np.random.default_rng(5)
eval_days = np.array(sorted(picks))
n_all = len(symbols)
sidx = {s: i for i, s in enumerate(symbols)}
didx = {pd.Timestamp(d): i for i, d in enumerate(eval_days)}
P = np.zeros((len(eval_days), n_all), bool)
for d, lst in picks.items():
    for s in lst:
        P[didx[d], sidx[s]] = True


def ci(x, B=3000):
    r = np.random.default_rng(1).integers(0, len(x), (B, len(x)))
    m = x[r].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


print(f"{'rule':16s} {'all 50':>9s} {'picked 10':>10s} {'CI picked':>18s} {'random-10 mean':>15s} {'picked-all (CI)':>24s}")
for kind, k, hold in [("live", 0.02, 60), ("live", 0.03, 60), ("live", 0.05, 120), ("live", 0.1, 120), ("live", 0.3, 90), ("candle5", 0.0, 0), ("candle15", 0.0, 0)]:
    sub = tr[(tr["kind"] == kind) & (tr["k"].round(3) == k) & (tr["hold"] == hold) & tr["date"].isin(eval_days)]
    M = np.zeros((len(eval_days), n_all))
    di = sub["date"].map(didx).to_numpy(); si = sub["symbol"].map(sidx).to_numpy()
    np.add.at(M, (di, si), sub["ret"].astype(float).to_numpy() - COST)
    allx = M.sum(1) / n_all * 1e4
    pk = (M * P).sum(1) / 10 * 1e4
    rnd = []
    for _ in range(300):
        order = np.argsort(rng.random(M.shape), axis=1)[:, :10]
        rnd.append((np.take_along_axis(M, order, axis=1).sum(1) / 10 * 1e4).mean())
    lo, hi = ci(pk); dl, dh = ci(pk - allx)
    print(f"{kind+'|'+str(k)+'|'+str(hold):16s} {allx.mean():9.2f} {pk.mean():10.2f} [{lo:6.2f},{hi:6.2f}] {np.mean(rnd):15.2f}  {np.mean(pk - allx):+6.2f} [{dl:+6.2f},{dh:+6.2f}]")
