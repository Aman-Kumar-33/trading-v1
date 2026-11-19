"""both_systems.exe - ONE program for both systems: the main 15m system (dashboard on port 8766) and the 5 minute system
(port 8767). Each keeps its own settings, database, logs and trading process; this program only starts both dashboards, opens
them in two browser tabs and keeps them running. The dashboards do the rest by themselves every trading day:

    after 15:30   the candle history of EVERY NIFTY 50 stock is loaded and saved for both systems, ready for the next morning
    09:10         both trading processes start (logged in and listening before the open); orders are allowed from 09:15
    15:00         every open position of both systems is sold (square-off), no buys after it

    both_systems.exe                 start both systems (if they are not running yet), open them in two browser tabs, keep them alive
    both_systems.exe --no-browser    the same without opening the browser
    both_systems.exe --stop          stop both dashboards (a trading process that is running keeps running)

Running it again while it is already running just opens the two tabs again.
Internally it starts itself once per system (`--only main` / `--only 5m`), hidden, so closing this window does not stop them.
"""
import argparse
import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

FROZEN = getattr(sys, "frozen", False)
if not FROZEN:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.core.config import ROOT as HERE   # the project folder: next to the exe, or the `second-system` folder beside/above it
SYSTEMS = {"main": {"name": "main system (15m)", "port": 8766}, "5m": {"name": "5 minute system", "port": 8767}}


def port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def child_command(key: str) -> list[str]:
    own = [sys.executable] if FROZEN else [sys.executable, str(Path(__file__).resolve())]
    return own + ["--only", key, "--no-browser"]


def spawn(key: str) -> None:
    env = {k: v for k, v in os.environ.items() if k != "ALGO_SETTINGS_FILE"}
    # a hidden console of its own (CREATE_NO_WINDOW), NOT "no console" (DETACHED_PROCESS): with no console every program the
    # dashboard starts - and it starts the trading process and checks on it all day - would open its own visible window
    flags = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW) if sys.platform == "win32" else 0
    subprocess.Popen(child_command(key), cwd=str(HERE), env=env, creationflags=flags, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)


def run_one(key: str) -> int:
    """This process IS one system's dashboard (what each `--only` child runs)."""
    if key == "5m":
        os.environ["ALGO_SETTINGS_FILE"] = "settings_5m.yaml"      # before anything reads the settings; the trading process inherits it
    else:
        os.environ.pop("ALGO_SETTINGS_FILE", None)
    sys.argv = [sys.argv[0], "--no-browser"]
    if not FROZEN:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dashboard import main as dashboard_main
    return dashboard_main()


def stop_all() -> int:
    if sys.platform != "win32":
        print("  --stop is for Windows; stop the processes by hand elsewhere")
        return 1
    # only the dashboards are called both_systems.exe: a trading process (python) is left alone and keeps trading
    r = subprocess.run(["taskkill", "/F", "/IM", "both_systems.exe", "/FI", f"PID ne {os.getpid()}"], capture_output=True, text=True,
                       creationflags=subprocess.CREATE_NO_WINDOW)
    print("  " + (r.stdout.strip() or r.stderr.strip() or "nothing to stop"))
    return 0


LOCK = HERE / "both_systems.lock"


def open_tabs() -> None:
    """The main system in one tab, the 5 minute system in a second tab of the same browser window."""
    webbrowser.open(f"http://127.0.0.1:{SYSTEMS['main']['port']}/")
    time.sleep(0.8)
    webbrowser.open_new_tab(f"http://127.0.0.1:{SYSTEMS['5m']['port']}/")


def another_running() -> bool:
    """True if a supervisor (another both_systems.exe, not a dashboard child) is already alive - then this one only opens the tabs."""
    from backend.execution.process_control import _is_running
    try:
        pid = int(LOCK.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False
    return pid != os.getpid() and _is_running(pid)


def supervise(open_browser: bool) -> int:
    if another_running():
        print("  both systems are already being kept running by another both_systems window - opening the two tabs")
        if open_browser:
            open_tabs()
        return 0
    try:
        LOCK.write_text(str(os.getpid()), encoding="utf-8")
    except OSError:
        pass
    print("  both systems - they start by themselves every trading day (history of all 50 stocks after 15:30, trading from 09:15, square-off 15:00)")
    first = True
    try:
        while True:
            for key, info in SYSTEMS.items():
                if not port_open(info["port"]):
                    print(f"  starting the {info['name']} (dashboard on http://127.0.0.1:{info['port']}/) ...")
                    spawn(key)
                    for _ in range(60):
                        if port_open(info["port"]):
                            break
                        time.sleep(0.5)
                    else:
                        print(f"  the {info['name']} did not come up - will try again")
            if first:
                first = False
                up = [i["name"] for i in SYSTEMS.values() if port_open(i["port"])]
                print(f"  running: {', '.join(up) or 'nothing'}.  15m system: http://127.0.0.1:{SYSTEMS['main']['port']}/   "
                      f"5 minute system: http://127.0.0.1:{SYSTEMS['5m']['port']}/")
                print("  Keep this window open to have a dashboard restarted automatically if it ever stops. "
                      "Closing it leaves them running; `both_systems.exe --stop` stops them.")
                if open_browser:
                    open_tabs()
            time.sleep(15)
    except KeyboardInterrupt:
        print("\n  left running in the background - `both_systems.exe --stop` stops them")
        return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", choices=list(SYSTEMS), help="(internal) run just this system's dashboard in this process")
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--stop", action="store_true", help="stop both dashboards")
    args = p.parse_args()
    if args.stop:
        return stop_all()
    if args.only:
        return run_one(args.only)
    return supervise(open_browser=not args.no_browser)


if __name__ == "__main__":
    raise SystemExit(main())
