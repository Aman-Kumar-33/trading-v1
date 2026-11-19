"""Launch the dashboard: starts the local HTTP server (backend/api/dashboard_server.py) and opens it in
your default browser. This is also the entry point packaged into Dashboard.exe - see
scripts/build_dashboard_exe.py. Run it from inside algo-system/ (same as every other script) so it finds
config/settings.yaml, data/, logs/ and backtest_data/.

    python scripts/dashboard.py              # http://127.0.0.1:8765, opens your browser
    python scripts/dashboard.py --port 9000 --no-browser
"""
import argparse
import threading
import webbrowser
from datetime import datetime

from _bootstrap import init

from backend.api.dashboard_server import make_server
from backend.data import chart_history
from backend.execution import process_control as pc
from backend.notify import telegram as tg
from backend.storage.db import init_db


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=None)   # default: instance.port in the settings (8766; the 5 minute system 8767)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--no-auto-start", action="store_true",
                   help="don't auto-start paper trading at market open while this dashboard runs")
    args = p.parse_args()
    settings, log = init("dashboard")
    inst = settings.section("instance")
    if args.port is None:
        args.port = int(inst.get("port", 8766))
    if not inst.get("auto_start", True):
        args.no_auto_start = True   # the 5 minute system only starts when its Start button is pressed
    settings.paths.ensure()
    init_db(settings.paths.db_file).close()   # make sure the trades/strategy_versions tables exist

    stop_event = threading.Event()
    if not args.no_auto_start:
        threading.Thread(target=pc.auto_start_loop, args=(settings, stop_event), daemon=True).start()

    # every stock's history of the trading timeframe, kept loaded in the background - and loaded again after each close, so the
    # next session starts from disk. The follower waits a little so the two systems do not hit Angel's rate limiter together.
    chart_history.start_warmer(settings, first_delay=90.0 if inst.get("follow") else 20.0)
    server = make_server(settings, port=args.port)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"  dashboard running at {url}  (Ctrl+C to stop)")
    log.info("dashboard started", extra={"data": {"url": url}})
    what = ("Paper trading auto-starts at 09:00 on trading days (your stock list, or today's top trending stocks "
            "when the box is empty)." if not args.no_auto_start else
            f"{inst.get('name') or 'This'} system does NOT start by itself - press Start Paper Trading on its page.")
    tg.notify(f"DASHBOARD SERVER STARTED ({inst.get('name') or 'main'}, port {args.port})\n{what}\n"
              f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")
    finally:
        stop_event.set()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
