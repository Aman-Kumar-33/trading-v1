"""Test the 5 minute system's MACD tolerance and hold time on many days of NIFTY 50 history (1-minute bars from
`scripts/fetch_research_1m.py`, replayed as pseudo-ticks - see backend/research/live_macd_replay.py).

    python scripts/tolerance_study.py                      # the whole grid, all downloaded days
    python scripts/tolerance_study.py --symbols MARUTI,SBIN --quick

Writes research_data/tolerance_study/trades.parquet (every simulated trade) and prints the summary tables.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.config import load_settings  # noqa: E402
from backend.data import trending  # noqa: E402
from backend.research import live_macd_replay as R  # noqa: E402

KS = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]
HOLDS = [30, 60, 90, 120, 180, 300]   # hold 0 is not tested: 1-minute bars cannot show sub-minute flicker (see the proxy check)
FLOOR = 0.02
SKIP_DAYS = 3            # first sessions of a series: the EMAs are still settling
MIN_ROWS = 358           # of the 360 real minutes (09:15-15:14): a session with more holes than this is left out


def stock_data(root: Path, sym: str, skip_days: int = SKIP_DAYS):
    df1 = R.load_1m(root / sym.replace("&", "_AND_"))
    if df1.empty:
        return None
    rows = df1[~df1["synthetic"]].groupby(df1["ts"].dt.date).size()
    good = set(rows[rows >= MIN_ROWS].index)
    dates = sorted(rows.index)
    skip = set(dates[:skip_days])
    return df1, good - skip


def collect(root: Path, symbols, ks, holds, floor, log=print, skip_days: int = SKIP_DAYS):
    frames = []   # one small frame per stock: (symbol, kind, k, hold, date, gross return)
    n_days = {}
    for i, sym in enumerate(symbols, 1):
        t0 = time.time()
        recs = []
        got = stock_data(root, sym, skip_days)
        if got is None:
            log(f"  {sym}: no data"); continue
        df1, ok_days = got
        bars5, bars15 = R.to_bars(df1, 5), R.to_bars(df1, 15)
        bs5, bs15 = R.bar_state(bars5), R.bar_state(bars15)
        tk = R.pseudo_ticks(df1, bs5, 5, floor, "auto")
        n_days[sym] = len(ok_days)
        for kind, bs, m in (("candle5", bs5, 5), ("candle15", bs15, 15)):
            for d, e, x, *_ in R.candle_close_trades(bs, df1, m):
                if d in ok_days:
                    recs.append((sym, kind, 0.0, 0, d, x / e - 1))
        for k in ks:
            for h in holds:
                for d, e, x, *_ in R.simulate(tk, k, h, floor):
                    day = tk.day_labels[d]
                    if day in ok_days:
                        recs.append((sym, "live", k, h, day, x / e - 1))
        frames.append(pd.DataFrame(recs, columns=["symbol", "kind", "k", "hold", "date", "ret"]).astype(
            {"symbol": "category", "kind": "category", "k": "float32", "hold": "int16", "ret": "float32"}))
        del df1, bars5, bars15, bs5, bs15, tk
        log(f"  [{i}/{len(symbols)}] {sym}: {len(ok_days)} days, {len(recs)} trades, {time.time() - t0:.0f}s")
    df = pd.concat(frames, ignore_index=True)
    return df, n_days


def boot_mean(per_day: pd.DataFrame, n_boot=2000, seed=1):
    """Mean net return per trade with a 95% interval, resampling whole DAYS (trades on one day are not independent)."""
    rng = np.random.default_rng(seed)
    s, c = per_day["sum"].to_numpy(), per_day["n"].to_numpy()
    if c.sum() == 0:
        return np.nan, np.nan, np.nan
    idx = rng.integers(0, len(s), (n_boot, len(s)))
    est = s[idx].sum(1) / np.maximum(c[idx].sum(1), 1)
    return s.sum() / c.sum(), np.percentile(est, 2.5), np.percentile(est, 97.5)


def summarise(trades: pd.DataFrame, all_days, cost, label_cols):
    out = []
    for key, g in trades.groupby(label_cols):
        net = g["ret"] - cost
        perday = pd.DataFrame({"sum": net.groupby(g["date"]).sum(), "n": net.groupby(g["date"]).size()}).reindex(all_days, fill_value=0)
        m, lo, hi = boot_mean(perday)
        out.append(dict(zip(label_cols, key if isinstance(key, tuple) else (key,)),
                        trades=len(g), per_stock_day=len(g) / (len(all_days) * trades["symbol"].nunique()),
                        win=100 * (net > 0).mean(), gross_bp=1e4 * g["ret"].mean(), net_bp=1e4 * m, lo_bp=1e4 * lo, hi_bp=1e4 * hi,
                        total_net_pct=100 * net.sum()))
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols")
    ap.add_argument("--quick", action="store_true", help="a smaller grid")
    ap.add_argument("--cost-bp", type=float, default=7.0, help="round-trip charges in basis points of the trade value")
    ap.add_argument("--skip-days", type=int, default=SKIP_DAYS, help="first sessions left out while the EMAs settle")
    args = ap.parse_args()
    settings = load_settings()
    root = settings.paths.root / "research_data" / "angel_1m"
    symbols = args.symbols.split(",") if args.symbols else trending.universe_symbols(settings)
    ks, holds = ([0.0, 0.02, 0.05, 0.1], [60, 120]) if args.quick else (KS, HOLDS)
    df, n_days = collect(root, symbols, ks, holds, FLOOR, skip_days=args.skip_days)
    out = settings.paths.root / "research_data" / "tolerance_study"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "trades.parquet")
    all_days = sorted(df["date"].unique())
    print(f"\n{len(symbols)} stocks, {len(all_days)} sessions ({all_days[0]} .. {all_days[-1]}), {len(df):,} simulated trades")
    s = summarise(df, all_days, args.cost_bp / 1e4, ["kind", "k", "hold"])
    pd.set_option("display.width", 220, "display.max_rows", 500)
    print(s.round(2).to_string(index=False))
    s.to_csv(out / "summary.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
