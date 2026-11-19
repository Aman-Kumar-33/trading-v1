"""Applies the Phase 3 walk-forward findings (2026-09-30/10-01) as strategy_vault status updates - the
"only strategies that passed the mass-backtest gates enter paper trading" rule from the plan, made
concrete. Never deletes a version or touches its params; only updates `strategy_versions.status` (and
the vault snapshot's meta.json) with a note citing the actual evidence, so the decision is auditable and
reversible - rerun scripts/walk_forward_strategies.py and re-run this script if new evidence changes the
call for any strategy.

Decisions, from backtest_data/results/walk_forward/:
  PAPER    quality_momentum_rank  - OOS +135,716 net P&L, 60% win rate, wfe=3.56 (OOS beat in-sample)
  RESEARCH (unchanged, needs more evidence before a decision):
    week52_high_momentum  - walk-forward test was invalid (1 OOS trade - test starved for data)
    time_series_momentum  - wfe=-10.21, unstable; aggregate OOS barely positive, don't trust it
    trend_ema_cross       - tuning regressed it to -27,883 vs its own untuned +26k baseline; likely
                            sample-size noise (only 15 symbols) picking an unstable stop multiplier,
                            not evidence the strategy itself is worse - rerun before deciding either way
  REJECTED (11 of 11 intraday strategies tested: zero profitable parameter combo in any of 15
  six-month windows across 3 years - see the plan's Phase 4 section for the full finding):
    candlestick_at_level, chart_pattern_breakout, vwap_trend_pullback, orb, vwap_reversion,
    supertrend_adx, price_action_structure, rsi_divergence, volume_spike_momentum, gap_fill_and_go,
    news_catalyst_momentum

    python scripts/promote_strategies.py            # apply all of the above
    python scripts/promote_strategies.py --dry-run  # show what would change, touch nothing
"""
import argparse
from datetime import datetime

from _bootstrap import init

from backend.storage.db import init_db
from backend.strategies import ALL_STRATEGIES
from backend.strategies.registry import get_status, load_params, register_version

VERSION = "v1"

PAPER = {
    "quality_momentum_rank": "Walk-forward OOS (2026-10-01, 15 symbols, 28 folds): +135,716 net P&L, "
    "60% win rate, walk_forward_efficiency=3.56 - OOS beat in-sample, a genuine validated edge.",
    "week52_high_momentum": "Full-system runs with live-parity costs: +87,338 net (3 symbols, 89 trades, "
    "2026-10-02) and +248,067 (30 symbols, 302 trades). Walk-forward sample too thin (5 OOS trades) - "
    "paper-trade to confirm.",
    "time_series_momentum": "Walk-forward OOS (2026-10-01, 30 symbols, 28 folds, Rs 2L tickets, next-open "
    "fills): +135,300 net on 156 trades, walk_forward_efficiency=3.7. Long-only since 2026-10-02.",
}
# Intraday: promoted to PAPER only - live data, paper money. The 2026-10-02 audit (386k signals) found gross
# edge below costs on average; these four had positive gross edge in both periods and trade only under the
# strongest measured condition (volume > 5x average) with hold-to-stop/square-off/carry exits. Paper
# results decide whether any of them ever gets LIVE status.
PAPER.update({
    sid: "Intraday, PAPER only (2026-10-02 edge audit): positive gross edge in discovery and holdout; trades only "
    "on volume > 5x average, no fixed target. Below cost on average in backtests - must prove itself on paper."
    for sid in ["gap_fill_and_go", "supertrend_adx"]
})

REJECTED = {
    sid: "2026-10-02 edge audit (386,519 signals, 30 stocks, 3 yrs, live-parity fills and costs): gross edge "
    "~0 or negative in both discovery and holdout under every condition tested. Rework needed, not tuning."
    for sid in ["candlestick_at_level", "chart_pattern_breakout", "vwap_trend_pullback", "vwap_reversion",
               "price_action_structure", "rsi_divergence", "news_catalyst_momentum"]
}
REJECTED.update({
    "orb": "2026-11-05 discard review: consistently negative/near-zero across every paper and backtest run "
    "since promotion (-5,037 live, -2,829 in the whole-system backtest) - no evidence of a real edge. v1 "
    "kept immutable in strategy_vault/orb/v1 if it needs re-evaluating later.",
    "volume_spike_momentum": "2026-11-05 discard review: worst single contributor in every system-level "
    "backtest and live paper trading (-3,834 live, -13,554 in the whole-system backtest). v1 kept immutable "
    "in strategy_vault/volume_spike_momentum/v1 if it needs re-evaluating later.",
})


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--set-live", help="comma separated strategy ids to move PAPER -> LIVE (real money). Your call, "
                                      "after reviewing their paper-trading results")
    p.add_argument("--set-paper", help="comma separated strategy ids to move back to PAPER (e.g. demote from LIVE)")
    args = p.parse_args()
    settings, log = init("promote_strategies")
    conn = init_db(settings.paths.db_file)
    by_id = {c.strategy_id: c for c in ALL_STRATEGIES}

    if args.set_live or args.set_paper:
        target, ids = ("LIVE", args.set_live) if args.set_live else ("PAPER", args.set_paper)
        for sid in [s.strip() for s in ids.split(",") if s.strip()]:
            current = get_status(conn, sid, VERSION)
            if sid not in by_id:
                print(f"  {sid!r}: unknown strategy")
                continue
            if target == "LIVE" and current != "PAPER":
                print(f"  {sid!r}: is {current}, not PAPER - only paper-trading strategies can go LIVE")
                continue
            print(f"  {sid:28s} {current} -> {target}")
            if not args.dry_run:
                strat = by_id[sid](params=load_params(sid, VERSION), version=VERSION)
                register_version(conn, strat, status=target,
                                 notes=f"set {target} manually {datetime.now():%Y-%m-%d %H:%M} (was {current})")
        conn.commit()
        conn.close()
        return 0

    changes = {**{sid: ("PAPER", note) for sid, note in PAPER.items()},
              **{sid: ("REJECTED", note) for sid, note in REJECTED.items()}}
    for sid, (status, note) in changes.items():
        cls = by_id.get(sid)
        if cls is None:
            print(f"  skipping {sid!r} - not found in backend.strategies.ALL_STRATEGIES")
            continue
        print(f"  {sid:28s} -> {status}")
        if args.dry_run:
            continue
        strat = cls(params=load_params(sid, VERSION), version=VERSION)
        register_version(conn, strat, status=status, notes=note)

    conn.commit()
    conn.close()
    if args.dry_run:
        print("\n  dry run - nothing written")
    else:
        print(f"\n  applied {len(changes)} status updates")
        log.info("strategy promotions applied", extra={"data": {"count": len(changes)}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
