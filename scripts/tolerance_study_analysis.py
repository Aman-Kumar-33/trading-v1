import sys
import numpy as np
import pandas as pd

ROOT = r"D:\Business\algo trading\v17 anthropic\second-system"
sys.path.insert(0, ROOT)
OUT = ROOT + r"\research_data\tolerance_study"
COST = 7e-4
pd.set_option("display.width", 230, "display.max_rows", 500, "display.max_columns", 50)

df = pd.read_parquet(OUT + r"\trades.parquet")
df["date"] = pd.to_datetime(df["date"])
days = np.array(sorted(df["date"].unique()))
n_stocks = df["symbol"].nunique()
print(f"{n_stocks} stocks, {len(days)} sessions {days[0].date()}..{days[-1].date()}, {len(df):,} trades")
df["cfg"] = df["kind"].astype(str) + "|" + df["k"].round(3).astype(str) + "|" + df["hold"].astype(str)
df["net"] = df["ret"].astype(float) - COST


def per_day(g, cost=COST):
    s = (g["ret"].astype(float) - cost).groupby(g["date"]).sum().reindex(days, fill_value=0.0)
    n = g.groupby("date").size().reindex(days, fill_value=0)
    return s.to_numpy(), n.to_numpy()


def boot(s, n, B=3000, seed=7):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), (B, len(s)))
    per_trade = s[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
    per_sd = s[idx].mean(1) / n_stocks
    return per_trade, per_sd


cfgs = df.groupby("cfg").size().index
rows = []
for c, g in df.groupby("cfg"):
    s, n = per_day(g)
    pt, psd = boot(s, n)
    rows.append(dict(cfg=c, kind=g["kind"].iloc[0], k=float(g["k"].iloc[0]), hold=int(g["hold"].iloc[0]), trades=int(n.sum()),
                     win=100 * (g["net"] > 0).mean(), gross_bp=1e4 * g["ret"].mean(),
                     net_bp_trade=1e4 * s.sum() / max(n.sum(), 1), t_lo=1e4 * np.percentile(pt, 2.5), t_hi=1e4 * np.percentile(pt, 97.5),
                     net_bp_stockday=1e4 * s.sum() / (len(days) * n_stocks), sd_lo=1e4 * np.percentile(psd, 2.5), sd_hi=1e4 * np.percentile(psd, 97.5),
                     p_pos=100 * (psd > 0).mean()))
S = pd.DataFrame(rows).sort_values(["kind", "k", "hold"]).reset_index(drop=True)
print("\n=== A. every rule: per-trade and per-stock-day net (bp = 0.01%), 95% CI by resampling days ===")
print(S.drop(columns="cfg").round(2).to_string(index=False))
S.to_csv(OUT + r"\summary_full.csv", index=False)

live = S[S.kind == "live"]
print("\n=== B. net bp per trade (rows k = tolerance x ATR, cols hold seconds) ===")
print(live.pivot(index="k", columns="hold", values="net_bp_trade").round(1).to_string())
print("\n=== C. net bp per stock-day (all rules' total, comparable across trade counts) ===")
print(live.pivot(index="k", columns="hold", values="net_bp_stockday").round(2).to_string())
print("\ntrades per stock-day:")
print((live.pivot(index="k", columns="hold", values="trades") / (len(days) * n_stocks)).round(2).to_string())

# ---- D. stability in three consecutive periods
print("\n=== D. net bp per trade in 3 consecutive periods (same dates for every rule) ===")
cuts = np.array_split(days, 3)
print("periods:", [f"{c[0].date()}..{c[-1].date()}" for c in cuts])
rows = []
for c, g in df.groupby("cfg"):
    r = {"cfg": c}
    for i, cut in enumerate(cuts):
        gg = g[g["date"].isin(cut)]
        r[f"p{i+1}_bp"] = 1e4 * gg["net"].mean() if len(gg) else np.nan
        r[f"p{i+1}_n"] = len(gg)
    rows.append(r)
D = pd.DataFrame(rows)
D["all_3_positive"] = (D[["p1_bp", "p2_bp", "p3_bp"]] > 0).all(axis=1)
print(D.round(1).to_string(index=False))

# ---- E. walk-forward: choose on the first half, test on the second half (and the reverse)
print("\n=== E. walk-forward: pick the best live setting on one half (by net bp per stock-day), score it on the other ===")
half = len(days) // 2
A, B = days[:half], days[half:]
live_cfgs = [c for c in cfgs if c.startswith("live")]


def score(c, ds, cost=COST):
    g = df[(df["cfg"] == c) & df["date"].isin(ds)]
    return 1e4 * (g["ret"].astype(float) - cost).sum() / (len(ds) * n_stocks), len(g)


for train, test, name in ((A, B, "first->second"), (B, A, "second->first")):
    tr = {c: score(c, train)[0] for c in live_cfgs}
    best = max(tr, key=tr.get)
    te = score(best, test)
    allte = np.array([score(c, test)[0] for c in live_cfgs])
    print(f"  {name}: best on train = {best} ({tr[best]:.2f} bp/stock-day) -> on test {te[0]:.2f} bp/stock-day "
          f"(test median over all live settings {np.median(allte):.2f}, best possible {allte.max():.2f}, worst {allte.min():.2f})")
a = np.array([score(c, A)[0] for c in live_cfgs]); b = np.array([score(c, B)[0] for c in live_cfgs])
print(f"  rank correlation of the 66 settings between the two halves: {pd.Series(a).corr(pd.Series(b), method='spearman'):.2f}")
cs5 = df[df.kind == "candle5"]; cs15 = df[df.kind == "candle15"]
for nm, g in (("candle5", cs5), ("candle15", cs15)):
    print(f"  {nm}: first half {1e4*g[g.date.isin(A)].net.sum()/(len(A)*n_stocks):.2f}  second half {1e4*g[g.date.isin(B)].net.sum()/(len(B)*n_stocks):.2f} bp/stock-day")

# ---- F. paired comparison with the candle-close rules (same days)
print("\n=== F. live setting minus candle-close rule, per stock-day, 95% CI by resampling days ===")
def daily_total(g):
    return (g["ret"].astype(float) - COST).groupby(g["date"]).sum().reindex(days, fill_value=0.0).to_numpy() / n_stocks
base5, base15 = daily_total(cs5), daily_total(cs15)
rng = np.random.default_rng(3)
idx = rng.integers(0, len(days), (4000, len(days)))
for c in ["live|0.02|60", "live|0.03|60", "live|0.05|60", "live|0.1|60", "live|0.05|120", "live|0.1|120"]:
    if c not in set(cfgs):
        continue
    x = daily_total(df[df.cfg == c])
    for nm, base in (("candle5", base5), ("candle15", base15)):
        d = x - base
        bs = d[idx].mean(1)
        print(f"  {c} - {nm}: {1e4*d.mean():+.2f} bp/stock-day  CI [{1e4*np.percentile(bs,2.5):+.2f}, {1e4*np.percentile(bs,97.5):+.2f}]")

# ---- G. cost / slippage sensitivity for the headline settings
print("\n=== G. net bp per stock-day under different round-trip costs ===")
for c in ["live|0.02|60", "live|0.03|60", "live|0.05|60", "live|0.1|60", "candle5|0.0|0", "candle15|0.0|0"]:
    if c not in set(cfgs):
        continue
    g = df[df.cfg == c]
    print(f"  {c:16s}", "  ".join(f"{cb} bp: {1e4*(g['ret'].astype(float)-cb/1e4).sum()/(len(days)*n_stocks):+.2f}" for cb in (0, 5, 7, 10, 15)))

# ---- H. which stocks: price tier / is the edge broad?
print("\n=== H. spread across stocks for live|0.03|60 and live|0.05|60: share of stocks with positive total net ===")
for c in ["live|0.03|60", "live|0.05|60", "candle5|0.0|0"]:
    if c not in set(cfgs):
        continue
    t = df[df.cfg == c].groupby("symbol", observed=True)["net"].sum() * 100
    print(f"  {c}: positive {int((t > 0).sum())}/{len(t)} stocks; median {t.median():.2f}%, best {t.max():.2f}% ({t.idxmax()}), worst {t.min():.2f}% ({t.idxmin()})")
