"""Download index constituent lists (NIFTY 50/100/200/500), the F&O ban list and circuit price bands,
then sync the instruments table.

    python scripts/build_universe.py
"""
from _bootstrap import init

from backend.data.nse_client import NSEClient
from backend.data.universe import refresh_index_lists, refresh_market_lists, sync_instruments
from backend.storage.db import init_db


def main() -> int:
    settings, log = init("build_universe")
    client = NSEClient()
    counts = refresh_index_lists(settings, client)
    counts.update(refresh_market_lists(settings, client))
    conn = init_db(settings.paths.db_file)
    counts["instruments_synced"] = sync_instruments(conn, settings)
    conn.close()
    log.info("universe updated", extra={"data": counts})
    for k, v in counts.items():
        print(f"  {k:20s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
