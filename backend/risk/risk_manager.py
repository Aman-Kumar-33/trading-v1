"""The risk manager: the one hard gate between an allocator-approved Signal and an actual order. Every
check here is a plain reject-or-pass rule, deliberately with no strategy-specific exceptions - this is
the place in the system that cannot be bypassed, per the plan's "Risk Manager (hard gate)" requirement.

Sizing is risk-% based (a stop-out loses ~`risk_pct_per_trade` x ensemble weight of the bucket), then
capped three ways - per-position notional, total open notional in the bucket (the 80/20 split), and a
share of the bar's traded volume - because risk-% sizing alone, on a tight stop, happily asks for a
position many times the capital you actually have. A trade is then rejected outright if Indian
transaction costs would eat more than `max_cost_to_risk_*` of the risk it's taking: with a 0.1-0.2% stop,
STT + exchange + stamp + GST + slippage alone are 0.2-0.4R per trade at ANY size, which no edge survives.

`RiskLimits.capital` is a placeholder - set it to your real trading capital in config before this is
ever wired to live money. Nothing in this module guesses that number.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, datetime, time

from backend.core.models import Horizon, Side, Signal
from backend.execution import cost_model as cm
from backend.risk.kill_switch import KillSwitch
from backend.risk.position_sizing import risk_based_qty


@dataclass
class RiskLimits:
    capital: float
    intraday_capital_pct: float = 0.8
    long_term_capital_pct: float = 0.2
    risk_pct_per_trade: float = 0.01
    max_concurrent_positions: int = 15
    daily_loss_limit_pct: float = 0.03
    max_consecutive_losses: int = 4
    no_entry_before: time = time(9, 20)      # first 5 min of the session blocked
    no_entry_after: time = time(14, 45)      # intraday only - mirrors strategies' own no_entry_after default
    intraday_leverage: float = 1.0           # MIS allows up to ~5x under SEBI peak-margin rules; 1x = none
    max_position_pct: float = 0.25           # one position's notional, as a share of its bucket's limit
    # Cost hurdle ("keep a margin over costs"): skip a trade if estimated round-trip fees exceed this share
    # of its rupee risk. 6,000 re-costed intraday trades: fees <=5% of risk netted +61k, <=10% ~breakeven,
    # <=25% (the old single limit) -2.2 lakh. Validate the exact level by walk-forward, not this sample.
    max_cost_to_risk_intraday: float = 0.10
    max_cost_to_risk_long_term: float = 0.25
    min_reward_to_cost: float = 3.0          # if the signal has a target: reward to target >= this x fees
    max_entries_per_symbol_per_day: int = 2  # intraday churn cap - re-entering after stop-outs pays fees again
    max_volume_participation: float = 0.05   # never more than 5% of the signal bar's traded volume
    allow_delivery_shorts: bool = False      # NSE cash segment: no overnight shorts without F&O/SLB


def not_executable_reason(signal: Signal, limits: RiskLimits | None = None) -> str | None:
    """Why a signal can't be placed as an order at all in the NSE cash segment, or None if it can.
    Shared with shadow tracking, so a strategy isn't credited (or blamed) for trades it could never take."""
    allow_shorts = limits.allow_delivery_shorts if limits is not None else False
    if signal.horizon == Horizon.LONG_TERM and signal.side == Side.SELL and not allow_shorts:
        return "delivery short selling isn't allowed in the NSE cash segment"
    return None


def pre_trade_check(signal: Signal, price: float, qty: int, limits: RiskLimits,
                    atr: float | None = None) -> tuple[str | None, float]:
    """The per-trade rules that don't depend on account state - shared by RiskManager.evaluate (live and
    the full-system backtest) and simulate_signals (per-strategy backtests / walk-forward), so research
    validates exactly the trades live would take. Returns (reject_reason or None, estimated round-trip cost)."""
    reason = not_executable_reason(signal, limits)
    if reason:
        return reason, 0.0
    if (signal.side == Side.BUY and signal.stop_loss >= price) or (signal.side == Side.SELL and signal.stop_loss <= price):
        return "stop already breached at the fill price", 0.0
    if qty <= 0:
        return "computed quantity is zero (degenerate stop distance)", 0.0
    est_cost = cm.round_trip_cost(price, price, qty, signal.side, signal.horizon).total
    if atr is not None and atr > 0:
        est_cost += cm.slippage(price, atr, qty)
    risk_amount = qty * abs(price - signal.stop_loss)
    limit = (limits.max_cost_to_risk_intraday if signal.horizon == Horizon.INTRADAY
             else limits.max_cost_to_risk_long_term)
    if est_cost > limit * risk_amount:
        return (f"fees would eat {est_cost / risk_amount:.0%} of the risk (limit {limit:.0%}) "
                f"- stop too tight for this ticket size"), est_cost
    if signal.targets:
        reward = qty * abs(signal.targets[0] - price)
        if reward < limits.min_reward_to_cost * est_cost:
            return (f"fees would eat {est_cost / reward:.0%} of the reward to target "
                    f"(need reward >= {limits.min_reward_to_cost:g}x fees)"), est_cost
    return None, est_cost


def widen_stop_for_cost_floor(signal: Signal, price: float, risk_capital: float, risk_pct: float,
                              limits: RiskLimits, atr: float | None = None, max_qty: int | None = None,
                              max_attempts: int = 10, max_stop_pct: float = 0.02) -> float:
    """A signal rejected only because its stop is too tight for its ticket size (cost-to-risk ratio) is a
    wasted opportunity, not a bad signal - the fix is a wider stop, not a rejection. The ratio is
    ~cost_rate x price / stop_distance (strictly decreasing in distance), so each failed attempt scales
    the distance up by exactly how far the ratio missed the limit by - this converges in 1-2 steps.
    `max_qty` should be the SAME exposure/volume cap the caller will apply afterwards - the flat
    per-order brokerage component means cost-to-risk also depends on qty, not just stop distance, so
    solving this against an uncapped qty and then capping qty afterwards can under-widen. Never widens
    past `max_stop_pct` of price, or past 90% of the distance to the signal's own target (so widening
    can't turn a trade into one that risks more than it could make) - the caller's own pre_trade_check
    has the final say regardless."""
    stop = signal.stop_loss
    max_dist = price * max_stop_pct
    if signal.targets:
        max_dist = min(max_dist, abs(signal.targets[0] - price) * 0.9)
    for _ in range(max_attempts):
        qty = risk_based_qty(risk_capital, risk_pct, price, stop)
        if max_qty is not None:
            qty = min(qty, max_qty)
        if qty <= 0:
            return stop
        reason, est_cost = pre_trade_check(replace(signal, stop_loss=stop), price, qty, limits, atr)
        if reason is None or "stop too tight for this ticket size" not in reason:
            return stop
        dist = abs(price - stop)
        risk_amount = qty * dist
        ratio = est_cost / risk_amount if risk_amount > 0 else float("inf")
        limit = limits.max_cost_to_risk_intraday if signal.horizon == Horizon.INTRADAY else limits.max_cost_to_risk_long_term
        new_dist = min(dist * (ratio / limit) * 1.05, max_dist)
        if new_dist <= dist:
            return stop
        stop = price - new_dist if signal.side == Side.BUY else price + new_dist
    return stop


@dataclass
class RiskState:
    """Mutable risk-manager state. `open_positions`/`open_notional` deliberately survive
    `reset_for_new_day` - long-term positions carry across days; only the daily counters reset."""
    trading_date: date | None = None
    realized_pnl_today: float = 0.0
    consecutive_losses: int = 0
    open_positions: int = 0
    open_notional: dict[str, float] = field(default_factory=lambda: {h.value: 0.0 for h in Horizon})
    intraday_entries_today: dict[str, int] = field(default_factory=dict)   # symbol -> approved entries
    kill_switch: KillSwitch = field(default_factory=KillSwitch)

    def reset_for_new_day(self, today: date) -> None:
        self.trading_date = today
        self.realized_pnl_today = 0.0
        self.consecutive_losses = 0
        self.intraday_entries_today.clear()
        self.kill_switch.reset()


@dataclass
class RiskDecision:
    approved: bool
    qty: int = 0
    reject_reason: str | None = None
    notional: float = 0.0
    est_cost: float = 0.0
    stop_loss: float = 0.0   # the stop actually sized against - may be wider than the signal's own stop
                             # (see widen_stop_for_cost_floor); callers must place the order at THIS stop,
                             # not signal.stop_loss, when approved


class RiskManager:
    def __init__(self, limits: RiskLimits, state: RiskState | None = None):
        self.limits = limits
        self.state = state or RiskState()
        # Set False by the live runner while the market-data feed is silent: pauses new entries only for
        # as long as the outage lasts (unlike the kill switch, which holds for the rest of the day).
        self.data_feed_ok = True

    # ------------------------------------------------------------------ state updates
    def on_trade_closed(self, pnl_net: float) -> None:
        """Call once per closed trade (win or loss) to keep the daily-loss and consecutive-loss
        counters current. Trips the kill switch immediately on breach - the next `evaluate()` call
        (even for the same tick/candle) will see it."""
        self.state.realized_pnl_today += pnl_net
        self.state.consecutive_losses = 0 if pnl_net > 0 else self.state.consecutive_losses + 1
        if self.state.realized_pnl_today <= -self.limits.capital * self.limits.daily_loss_limit_pct:
            self.state.kill_switch.trip("daily loss limit")
        if self.state.consecutive_losses >= self.limits.max_consecutive_losses:
            self.state.kill_switch.trip("max consecutive losses")

    def on_position_opened(self, horizon: Horizon, notional: float) -> None:
        self.state.open_positions += 1
        self.state.open_notional[horizon.value] += notional

    def on_position_closed(self, horizon: Horizon, notional: float) -> None:
        self.state.open_positions = max(0, self.state.open_positions - 1)
        self.state.open_notional[horizon.value] = max(0.0, self.state.open_notional[horizon.value] - notional)

    # ------------------------------------------------------------------ sizing
    def bucket_limit(self, horizon: Horizon) -> float:
        """Max total open notional for a bucket: its share of capital x allowed leverage."""
        if horizon == Horizon.INTRADAY:
            return self.limits.capital * self.limits.intraday_capital_pct * self.limits.intraday_leverage
        return self.limits.capital * self.limits.long_term_capital_pct

    def _risk_capital(self, horizon: Horizon) -> float:
        pct = self.limits.intraday_capital_pct if horizon == Horizon.INTRADAY else self.limits.long_term_capital_pct
        return self.limits.capital * pct

    def nominal_qty(self, signal: Signal, price: float, weight: float = 1.0) -> int:
        """Quantity at this weight, with the per-position cap but ignoring what's currently open - what a
        signal would be sized at if taken alone. Shadow tracking uses this so costs (which depend on
        ticket size) are measured realistically."""
        risk_qty = risk_based_qty(self._risk_capital(signal.horizon), self.limits.risk_pct_per_trade * weight,
                                  price, signal.stop_loss)
        position_cap = int(self.bucket_limit(signal.horizon) * self.limits.max_position_pct / price) if price > 0 else 0
        return max(0, min(risk_qty, position_cap))

    # ------------------------------------------------------------------ the gate
    def evaluate(self, signal: Signal, now: datetime, current_price: float, *, weight: float = 1.0,
                 bar_volume: float | None = None, atr: float | None = None) -> RiskDecision:
        """`current_price` is the price the order is expected to FILL at (next tick / next bar open) -
        sizing always uses it, never the strategy's own reference `signal.entry`, because a market order
        fills wherever the market is. `weight` is the allocator's 0-1 ensemble weight and scales the risk
        taken. Never raises; every rejection is a `RiskDecision(approved=False, reject_reason=...)`."""
        if self.state.kill_switch.tripped:
            return RiskDecision(False, reject_reason=f"kill switch: {self.state.kill_switch.reason}")
        if not self.data_feed_ok:
            return RiskDecision(False, reject_reason="market data feed stale - entries paused")
        reason = not_executable_reason(signal, self.limits)
        if reason:
            return RiskDecision(False, reject_reason=reason)
        if self.state.open_positions >= self.limits.max_concurrent_positions:
            return RiskDecision(False, reject_reason="max concurrent positions reached")
        intraday = signal.horizon == Horizon.INTRADAY
        if intraday:
            t = now.time()
            if t < self.limits.no_entry_before:
                return RiskDecision(False, reject_reason="no-entry window at session open")
            if t >= self.limits.no_entry_after:
                return RiskDecision(False, reject_reason="no-entry window before square-off")
            if self.state.intraday_entries_today.get(signal.symbol, 0) >= self.limits.max_entries_per_symbol_per_day:
                return RiskDecision(False, reject_reason="max entries per symbol per day reached")
        if weight <= 0:
            return RiskDecision(False, reject_reason="zero ensemble weight")
        price = current_price
        if (signal.side == Side.BUY and signal.stop_loss >= price) or (signal.side == Side.SELL and signal.stop_loss <= price):
            return RiskDecision(False, reject_reason="stop already breached at the fill price")

        limit = self.bucket_limit(signal.horizon)
        caps = {
            "per-position notional cap": int(limit * self.limits.max_position_pct / price),
            "no capital left in the bucket": int(max(0.0, limit - self.state.open_notional[signal.horizon.value]) / price),
        }
        if bar_volume is not None:
            caps["bar volume participation cap"] = int(bar_volume * self.limits.max_volume_participation)

        risk_capital = self._risk_capital(signal.horizon)
        risk_pct = self.limits.risk_pct_per_trade * weight
        stop = widen_stop_for_cost_floor(signal, price, risk_capital, risk_pct, self.limits, atr,
                                         max_qty=min(caps.values()))

        risk_qty = risk_based_qty(risk_capital, risk_pct, price, stop)
        if risk_qty <= 0:
            return RiskDecision(False, reject_reason="computed quantity is zero (degenerate stop distance)")
        qty = min(risk_qty, *caps.values())
        if qty <= 0:
            binding = min(caps, key=caps.get)
            return RiskDecision(False, reject_reason=f"quantity capped to zero - {binding}")

        reason, est_cost = pre_trade_check(replace(signal, stop_loss=stop), price, qty, self.limits, atr)
        if reason:
            return RiskDecision(False, reject_reason=reason)
        if intraday:
            self.state.intraday_entries_today[signal.symbol] = self.state.intraday_entries_today.get(signal.symbol, 0) + 1
        return RiskDecision(True, qty=qty, notional=qty * price, est_cost=est_cost, stop_loss=stop)
