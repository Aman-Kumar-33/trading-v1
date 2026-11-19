"""Seed v1 configs, strategy_vault/ snapshots and DB registration for every strategy in the pool.

Run once (or whenever a brand-new strategy is added to backend/strategies/__init__.py) to give every
strategy a starting "v1" version with its class defaults as params, status RESEARCH (nothing here is
paper-traded or live - that only happens after Phase 3 backtest gates pass).

    python scripts/init_strategy_vault.py
    python scripts/init_strategy_vault.py --strategy orb   # just one
"""
import argparse

from _bootstrap import init

from backend.storage.db import init_db
from backend.strategies import ALL_STRATEGIES
from backend.strategies.registry import list_versions, load_params, register_version, save_params, snapshot


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", help="only seed this strategy_id")
    p.add_argument("--vault-root", default="../strategy_vault")
    args = p.parse_args()
    settings, log = init("init_strategy_vault")
    vault_root = settings.resolve(args.vault_root)
    conn = init_db(settings.paths.db_file)

    seeded, skipped = 0, 0
    for cls in ALL_STRATEGIES:
        if args.strategy and cls.strategy_id != args.strategy:
            continue
        if "v1" in list_versions(cls.strategy_id):
            print(f"  {cls.strategy_id:28s} v1 config already exists - skipping config+snapshot")
            skipped += 1
        else:
            save_params(cls.strategy_id, "v1", dict(cls.default_params))
            strat = cls(params=load_params(cls.strategy_id, "v1"), version="v1")
            try:
                snapshot(strat, vault_root, status="RESEARCH", notes="initial seed from class defaults (Phase 2)")
            except FileExistsError:
                pass
            print(f"  {cls.strategy_id:28s} seeded v1 ({cls.horizon.value}, {len(cls.default_params)} params)")
            seeded += 1
        strat = cls(params=load_params(cls.strategy_id, "v1"), version="v1")
        register_version(conn, strat, status="RESEARCH", notes="Phase 2 seed")

    conn.close()
    log.info("strategy vault initialized", extra={"data": {"seeded": seeded, "skipped": skipped}})
    print(f"done: {seeded} seeded, {skipped} already present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
