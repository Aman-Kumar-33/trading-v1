"""Launch the "5 minute" system: a second dashboard (port 8767) with its own paper-trading process, database, logs
and history cache, configured by config/settings_5m.yaml. It trades the stocks the main system trades, from live
ticks, on 5 minute candles with a MACD tolerance - and it never starts trading by itself.

    python scripts/dashboard_5m.py              # http://127.0.0.1:8767
    python scripts/dashboard_5m.py --no-browser

This is also the entry point packaged into second_dashboard_5m.exe.
"""
import os
import sys

os.environ.setdefault("ALGO_SETTINGS_FILE", "settings_5m.yaml")   # before anything reads the settings; the trading
                                                                    # process it launches inherits it

from dashboard import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
