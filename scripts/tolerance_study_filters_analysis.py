"""Score the whipsaw filters: choose on the first 60% of the sessions, confirm on the last 40% (never used for choosing)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research_data" / "tolerance_study"
COST = 7e-4
pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_columns", 40)

df = pd.read_parquet(OUT / "filter_trades.parquet")
df["date"] = pd.to_datetime(df["date"])
df["symbol"] = df["symbol"].astype(str)
df["cand"] = df["cand"].astype(str)
df["k"] = df["k"].astype(float).round(3)
days = np.array(sorted(df["date"].unique()))
n_stocks = df["symbol"].nunique()
cut = int(len(days) * 0.6)
TRAIN, TEST = days[:cut], days[cut:]
print(f"{n_stocks} stocks, {len(days)} sessions; train {TRAIN[0].date()}..{TRAIN[-1].date()} ({len(TRAIN)}), test {TEST[0].date()}..{TEST[-1].date()} ({len(TEST)})")


def daily(g, ds):
    s = (g["ret"].astype(float) - COST).groupby(g["date"]).sum().reindex(ds, fill_value=0.0).to_numpy()
    n = g.groupby("date").size().reindex(ds, fill_value=0).to_numpy()
    return s, n


def stats(g, ds):
    s, n = daily(g, ds)
    gg = g[g["date"].isin(ds)]
    return dict(trades_sd=n.sum() / (len(ds) * n_stocks), gross=1e4 * gg["ret"].astype(float).mean() if len(gg) else np.nan,
                net_trade=1e4 * s.sum() / max(n.sum(), 1), net_sd=1e4 * s.sum() / (len(ds) * n_stocks))


def boot_diff(a, b, B=4000):
    d = (a - b) / n_stocks * 1e4
    r = np.random.default_rng(11).integers(0, len(d), (B, len(d)))
    m = d[r].mean(1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


rows = []
for (k, h), g in df.groupby(["k", "hold"]):
    base = g[g["cand"] == "base"]
    bt, bte = daily(base, TRAIN)[0], daily(base, TEST)[0]
    for cand, c in g.groupby("cand"):
        tr, te = stats(c, TRAIN), stats(c, TEST)
        st, ste = daily(c, TRAIN)[0], daily(c, TEST)[0]
        d_tr = boot_diff(st, bt)
        d_te = boot_diff(ste, bte)
        rows.append(dict(k=k, hold=h, cand=cand, tr_trades=tr["trades_sd"], tr_gross=tr["gross"], tr_net_sd=tr["net_sd"],
                         tr_vs_base=d_tr[0], te_trades=te["trades_sd"], te_gross=te["gross"], te_net_sd=te["net_sd"],
                         te_vs_base=d_te[0], te_lo=d_te[1], te_hi=d_te[2]))
R_ = pd.DataFrame(rows)
R_.to_csv(OUT / "filter_summary.csv", index=False)

for (k, h), g in R_.groupby(["k", "hold"]):
    print(f"\n=========== base: live, tolerance {k} x ATR, hold {h}s ===========  (net = bp per stock-day after 7 bp charges; vs_base = gain over this base)")
    g = g.sort_values("tr_net_sd", ascending=False)
    print(g.drop(columns=["k", "hold"]).round(2).to_string(index=False))

print("\n=========== the confirmation: the best 5 on TRAIN (per base), judged on TEST only ===========")
for (k, h), g in R_.groupby(["k", "hold"]):
    top = g[g["cand"] != "base"].sort_values("tr_net_sd", ascending=False).head(5)
    base = g[g["cand"] == "base"].iloc[0]
    print(f"\nbase k={k} hold={h}: TEST net of the base = {base.te_net_sd:.2f} bp/stock-day")
    for _, r in top.iterrows():
        print(f"   {r.cand:42s} train {r.tr_net_sd:7.2f} -> TEST {r.te_net_sd:7.2f}  gain over base on TEST {r.te_vs_base:+6.2f} bp "
              f"[{r.te_lo:+.2f}, {r.te_hi:+.2f}]   trades/stock-day {r.te_trades:.2f}")
