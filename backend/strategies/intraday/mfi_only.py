"""MFI-only: BUY when MFI drops below 20 then later crosses back above 20 (oversold recovery). SELL/exit
when MFI rises above 80 then later crosses back below 80 (overbought recovery). Stateful across the
series (not a fixed lookback window) - an "armed" flag set when the threshold is breached, cleared once
the recovery-cross fires."""
from __future__ import annotations

from backend.core.models import Horizon, Side, Signal
from backend.strategies.base import Strategy, StrategyContext


class MfiOnly(Strategy):
    strategy_id = "mfi_only"
    horizon = Horizon.INTRADAY
    needs_full_features = False
    default_params = {"oversold": 20.0, "overbought": 80.0, "wide_stop_atr_mult": 10.0}

    def generate_signals(self, ctx: StrategyContext) -> list[Signal]:
        df = ctx.df
        p = self.params
        out: list[Signal] = []
        armed_buy, armed_sell = False, False
        for i in range(1, len(df)):
            row = df.iloc[i]
            mfi, prev_mfi = row["mfi_14"], df["mfi_14"].iloc[i - 1]
            if mfi != mfi or prev_mfi != prev_mfi:
                continue
            if mfi < p["oversold"]:
                armed_buy = True
            if mfi > p["overbought"]:
                armed_sell = True
            side = None
            if armed_buy and prev_mfi <= p["oversold"] < mfi:
                armed_buy = False
                # quality filter on entries only: require above-average volume on the actual cross bar
                vol, avg_vol = row.get("volume"), row.get("avg_volume")
                if avg_vol == avg_vol and vol is not None and vol <= avg_vol:
                    continue
                side, reason = Side.BUY, "MFI recovered from oversold"
            elif armed_sell and prev_mfi >= p["overbought"] > mfi:
                side, reason = Side.SELL, "MFI recovered from overbought"
                armed_sell = False
            if side is None:
                continue
            entry = float(row["close"])
            atr = row.get("atr_14") or entry * 0.02
            stop = entry - p["wide_stop_atr_mult"] * atr if side == Side.BUY else entry + p["wide_stop_atr_mult"] * atr
            out.append(Signal(symbol=ctx.symbol, side=side, horizon=self.horizon, stop_loss=stop,
                              strategy_id="", strategy_version="", ts=row["ts"], timeframe=ctx.timeframe,
                              entry=entry, reason=reason))
        return out
