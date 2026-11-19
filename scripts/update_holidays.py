"""Refresh config/nse_holidays.yaml from NSE's official holiday list.

    python scripts/update_holidays.py
"""
from _bootstrap import init

from backend.data.market_calendar import refresh_holidays_file
from backend.data.nse_client import NSEClient


def main() -> int:
    settings, log = init("update_holidays")
    path = settings.resolve(settings.section("market").get("holidays_file", "config/nse_holidays.yaml"))
    fetched = NSEClient().holidays()
    n = refresh_holidays_file(path, fetched)
    log.info(f"holiday file updated: {n} holidays", extra={"data": {"path": str(path)}})
    for h in fetched:
        print(f"  {h['date']}  {h['description']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
