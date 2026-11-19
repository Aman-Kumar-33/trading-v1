"""Hybrid: entry only via the MACD bullish cross (same condition as macd_only's BUY). Exit signals are
emitted independently whenever either condition fires - a MACD bearish cross, or MFI's overbought-
recovery (crosses above 80, later back below 80, same condition as mfi_only's SELL) - each tagged with a
distinct `reason`. This strategy does NOT decide whether to act on an exit signal immediately: it has no
view of the real entry price/qty/costs, so that decision (act on the first exit signal only if the
position is already profitable after costs, otherwise wait for the second, different exit signal to force
it) lives in strategy_runner.py's `_act`, which has the real position. Not tracking an internal
"in position" flag for entries is deliberate and safe: the runner's own `has_position` check already
prevents a duplicate real entry while one is open (see macd_only.py / mfi_only.py, which work the same
way)."""
from __future__ import annotations

from backend.core.models import Horizon, Side, Signal
from backend.strategies.base import Strategy, StrategyContext


class MacdMfiHybrid(Strategy):
    strategy_id = "macd_mfi_hybrid"
    horizon = Horizon.INTRADAY
    needs_full_features = False
    default_params = {"overbought": 80.0, "wide_stop_atr_mult": 10.0}

    def generate_signals(self, ctx: StrategyContext) -> list[Signal]:
        df = ctx.df
        p = self.params
        out: list[Signal] = []
        armed_sell = False
        for i in range(1, len(df)):
            row = df.iloc[i]
            macd, sig, mfi = row["macd"], row["macd_signal"], row["mfi_14"]
            prev_macd, prev_sig = df["macd"].iloc[i - 1], df["macd_signal"].iloc[i - 1]
            prev_mfi = df["mfi_14"].iloc[i - 1]
            if any(v != v for v in (macd, sig, prev_macd, prev_sig)):
                continue
            if mfi == mfi and mfi > p["overbought"]:
                armed_sell = True
            bull = prev_macd <= prev_sig and macd > sig
            bear = prev_macd >= prev_sig and macd < sig
            fires: list[tuple] = []
            if bull:
                # quality filter on entries only: histogram expanding (not a bare crossover) + above-
                # average volume - same rule as macd_only, applied only to the entry, never the exit.
                hist, prev_hist = row.get("macd_hist"), df["macd_hist"].iloc[i - 1]
                vol, avg_vol = row.get("volume"), row.get("avg_volume")
                expanding = hist == hist and prev_hist == prev_hist and abs(hist) > abs(prev_hist)
                vol_ok = not (avg_vol == avg_vol and vol is not None and vol <= avg_vol)
                if expanding and vol_ok:
                    fires.append((Side.BUY, "MACD cross entry"))
            if bear:
                fires.append((Side.SELL, "exit: MACD bear cross"))
            if armed_sell and mfi == mfi and prev_mfi == prev_mfi and prev_mfi >= p["overbought"] > mfi:
                armed_sell = False
                fires.append((Side.SELL, "exit: MFI recovered from overbought"))
            for side, reason in fires:
                entry = float(row["close"])
                atr = row.get("atr_14") or entry * 0.02
                stop = (entry - p["wide_stop_atr_mult"] * atr if side == Side.BUY
                       else entry + p["wide_stop_atr_mult"] * atr)
                out.append(Signal(symbol=ctx.symbol, side=side, horizon=self.horizon, stop_loss=stop,
                                  strategy_id="", strategy_version="", ts=row["ts"], timeframe=ctx.timeframe,
                                  entry=entry, reason=reason))
        return out
