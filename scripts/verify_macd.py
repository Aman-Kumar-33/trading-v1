"""Check the MACD signals against real Angel One bars, one symbol at a time - for eyeballing against a chart.

For each symbol this fetches the same continuous 30m series the live system warms up on (and refreshes the
on-disk cache the dashboard chart reads), then prints every MACD cross in the last N sessions: bar start
time (IST), BUY/SELL, the bar's close, and the MACD / signal values. The same crosses are recomputed with an
independent, TradingView-style implementation (EMAs seeded with an SMA, written without pandas) and the two
are compared - they must agree on every cross, which proves the system's numbers aren't a pandas artefact.

    python scripts/verify_macd.py RELIANCE HDFCBANK            # last 5 sessions
    python scripts/verify_macd.py RELIANCE --sessions 10
    python scripts/verify_macd.py RELIANCE --offline           # use the cached series, no login

How to read a chart against it: a cross is stamped with the START time of the bar whose CLOSE completes it
(a 09:15 bar closes at 09:45). On a chart, BUY = the MACD line is below the signal line on the previous bar
and above it on this one.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from _bootstrap import init

from backend.core.clock import IST
from backend.data import history_store
from backend.features import indicators as ind
from backend.strategies.intraday.macd_only import macd_cross_masks


def reference_macd(close: np.ndarray, fast: int = 12, slow: int = 26, sig: int = 9) -> tuple[np.ndarray, np.ndarray]:
    """TradingView / TA-Lib convention: every EMA starts from the SMA of its first `n` inputs."""
    def ema(values: np.ndarray, n: int) -> np.ndarray:
        out = np.full(len(values), np.nan)
        first = int(np.argmax(~np.isnan(values)))           # first usable input
        if len(values) - first < n:
            return out
        k = 2.0 / (n + 1)
        out[first + n - 1] = values[first:first + n].mean()
        for i in range(first + n, len(values)):
            out[i] = values[i] * k + out[i - 1] * (1 - k)
        return out
    macd = ema(close, fast) - ema(close, slow)
    return macd, ema(macd, sig)


def crosses(ts: pd.Series, close: pd.Series, macd: np.ndarray, signal: np.ndarray) -> pd.DataFrame:
    bull, bear = macd_cross_masks(macd, signal)
    idx = np.flatnonzero(bull | bear)
    return pd.DataFrame({"bar_start": ts.iloc[idx].dt.strftime("%Y-%m-%d %H:%M").to_numpy(),
                         "cross": np.where(bull[idx], "BUY", "SELL"), "close": close.iloc[idx].to_numpy(),
                         "macd": macd[idx].round(4), "signal": signal[idx].round(4)})


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("symbols", nargs="+")
    p.add_argument("--sessions", type=int, default=5, help="how many recent trading days to list")
    p.add_argument("--timeframe", default=None)
    p.add_argument("--offline", action="store_true", help="use the cached series instead of fetching from Angel One")
    args = p.parse_args()
    settings, _ = init("verify_macd")
    tf = args.timeframe or settings.section("trading").get("intraday_timeframe", "15m")
    days = int(settings.section("trading").get("history_days", 60))
    symbols = [s.upper() for s in args.symbols]

    angel = symbol_map = None
    if not args.offline:
        from backend.core.events import EventBus
        from backend.brokers.angel import AngelOneAdapter
        from backend.core.secrets import get_angel_credentials
        from backend.data import angel_instruments as ai
        creds = get_angel_credentials()
        angel = AngelOneAdapter(EventBus(), creds.api_key, creds.client_code, creds.pin, creds.totp_secret)
        angel.login()
        master_cfg = settings.section("angelone")
        master = ai.load_instrument_master(settings.paths.universe / "angel_instrument_master.json",
                                           master_cfg["instrument_master_url"], master_cfg.get("instrument_master_refresh_hours", 24))
        symbol_map = ai.build_symbol_map(master, exchange="NSE")
        time.sleep(3)   # Angel rejects the first data call right after login

    all_ok = True
    for n, sym in enumerate(symbols):
        if angel is not None:
            if sym not in symbol_map:
                print(f"\n{sym}: not in Angel One's instrument master"); all_ok = False; continue
            if n:
                time.sleep(history_store.CALL_SPACING_SECONDS)
            df = history_store.refresh_symbol(settings, angel, sym, symbol_map[sym], tf, days, datetime.now(IST))
        else:
            df = history_store.load_history(settings, sym, tf)
            if df is None:
                print(f"\n{sym}: no cached series - run without --offline first"); all_ok = False; continue
        close = df["close"].astype(float)
        m = ind.macd(close)
        ours = crosses(df["ts"], close, m["macd"].to_numpy(), m["signal"].to_numpy())
        ref_macd, ref_sig = reference_macd(close.to_numpy())
        ref = crosses(df["ts"], close, ref_macd, ref_sig)

        recent_days = sorted(df["ts"].dt.date.unique())[-args.sessions:]
        cutoff = f"{recent_days[0]} 00:00"
        shown, shown_ref = ours[ours["bar_start"] >= cutoff], ref[ref["bar_start"] >= cutoff]
        same = shown[["bar_start", "cross"]].reset_index(drop=True).equals(shown_ref[["bar_start", "cross"]].reset_index(drop=True))
        all_ok &= same
        sessions_bars = df["ts"].dt.date.value_counts().sort_index()
        print(f"\n=== {sym} {tf}: {len(df)} bars {df['ts'].iloc[0]:%Y-%m-%d} -> {df['ts'].iloc[-1]:%Y-%m-%d %H:%M}, "
              f"{int(sessions_bars.iloc[:-1].tail(3).median())} bars per completed day (13 = full day incl. the 15:15 closing bar)")
        print(shown.to_string(index=False) if len(shown) else "  no crosses in this window")
        print(f"  independent implementation agrees on all {len(shown)} crosses: {'YES' if same else 'NO'}")
        if not same:
            print("  reference says:\n" + shown_ref.to_string(index=False))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
