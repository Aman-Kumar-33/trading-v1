"""Start/stop scripts/run_trading.py as a subprocess for the dashboard's Paper/Live pages. Tracked via
data/state/trading_process_<mode>.json so a dashboard restart or browser refresh still shows the real
state - the source of truth is the OS process (checked by PID), not anything held in server memory.

Stopping asks nicely first (SIGINT on POSIX, CTRL_BREAK_EVENT on Windows - run_trading.py already catches
KeyboardInterrupt and saves state before exiting), and falls back to a hard kill if the process ignores
that within `grace_seconds`.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from backend.core.config import Settings

_launch_lock = threading.Lock()   # the dashboard server is threaded - two near-simultaneous start
                                   # requests could both see "not running" before either writes state


def _state_path(settings: Settings, mode: str) -> Path:
    return settings.paths.live_data / "state" / f"trading_process_{mode}.json"


def _windows_pid_alive(pid: int) -> bool:
    """True while a process with this id exists and has not exited - a direct Windows API call (OpenProcess + exit code), no
    child program, so nothing flashes on screen however often the dashboard asks."""
    import ctypes
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    handle = kernel32.OpenProcess(0x1000, False, int(pid))            # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return ctypes.get_last_error() == 5                           # access denied: it exists, we just may not look inside
    try:
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return True
        return code.value == 259                                      # STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def _is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        return _windows_pid_alive(pid)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def status(settings: Settings, mode: str) -> dict:
    path = _state_path(settings, mode)
    if not path.exists():
        return {"running": False}
    info = json.loads(path.read_text(encoding="utf-8"))
    info["running"] = _is_running(info.get("pid", -1))
    return info


def _launch(settings: Settings, mode: str, args: list[str]) -> dict:
    """The actual subprocess launch, with `args` fully formed by the caller - split out so tests can
    launch a harmless stand-in script instead of the real trading process."""
    with _launch_lock:
        current = status(settings, mode)
        if current.get("running"):
            return current
        kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if sys.platform == "win32" else {}
        proc = subprocess.Popen(args, cwd=str(settings.paths.root), **kwargs)
        info = {"running": True, "pid": proc.pid, "mode": mode, "started_at": datetime.now().isoformat()}
        path = _state_path(settings, mode)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(info), encoding="utf-8")
        return info


def _python_for_scripts(settings: Settings) -> str:
    """sys.executable IS Dashboard.exe when frozen (PyInstaller) - scripts/run_trading.py isn't bundled
    into it, so launching `[sys.executable, "run_trading.py", ...]` just re-runs the dashboard exe with
    nonsense args, which exits immediately (the "starts, errors, stops" bug). Use the project's own venv
    interpreter instead, next to the exe - it already has every dependency run_trading.py needs."""
    if not getattr(sys, "frozen", False):
        return sys.executable
    venv_py = settings.paths.root / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if venv_py.exists():
        return str(venv_py)
    import shutil
    return shutil.which("python") or shutil.which("python3") or sys.executable


def start(settings: Settings, mode: str, symbols: str | None = None) -> dict:
    args = [_python_for_scripts(settings), str(settings.paths.root / "scripts" / "run_trading.py"), "--mode", mode]
    if mode == "live":
        args.append("--i-accept-real-money")
    if symbols:
        args += ["--symbols", symbols]
    return _launch(settings, mode, args)


def stop(settings: Settings, mode: str, grace_seconds: float = 5.0) -> dict:
    current = status(settings, mode)
    if not current.get("running"):
        return current
    pid = current["pid"]
    try:
        if sys.platform == "win32":
            os.kill(pid, signal.CTRL_BREAK_EVENT)
        else:
            os.kill(pid, signal.SIGINT)
    except OSError:
        pass
    deadline = time.time() + grace_seconds
    while time.time() < deadline and _is_running(pid):
        time.sleep(0.2)
    if _is_running(pid):
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        else:
            os.kill(pid, signal.SIGKILL)
    info = {"running": False, "pid": pid, "mode": mode, "stopped_at": datetime.now().isoformat()}
    _state_path(settings, mode).write_text(json.dumps(info), encoding="utf-8")
    return info


COMMAND_MAX_AGE_SECONDS = 6 * 3600     # a force-exit queued hours ago (nothing was running to take it) is not acted on later


def _commands_path(settings: Settings, mode: str) -> Path:
    """The old single-file queue (read once if it is still there)."""
    return settings.paths.live_data / "state" / f"commands_{mode}.json"


def _commands_dir(settings: Settings, mode: str) -> Path:
    return settings.paths.live_data / "state" / f"commands_{mode}"


def queue_close_position(settings: Settings, mode: str, symbol: str) -> None:
    """Queues a manual force-exit for the running trading process of this mode (it polls once a second in its main loop). One
    small file per command, written atomically: the dashboard adding a command and the trading process taking commands can no
    longer overwrite each other (the old read-modify-write of one shared file could lose a force-exit)."""
    import uuid
    folder = _commands_dir(settings, mode)
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{time.time_ns()}-{uuid.uuid4().hex[:8]}"
    tmp = folder / f"{name}.tmp"
    tmp.write_text(json.dumps({"action": "close_position", "symbol": symbol, "queued_at": datetime.now().isoformat(timespec="seconds")}),
                   encoding="utf-8")
    tmp.replace(folder / f"{name}.json")
    _legacy_append(settings, mode, symbol)


def _legacy_append(settings: Settings, mode: str, symbol: str) -> None:
    """Trading processes started before the per-file queue existed (18 Nov 2026, until they exit at the close) read only the old
    single `commands_<mode>.json`. Their force-exit still has to work, so the command is also left there; a process that reads both
    just closes the stock once and ignores the second copy (there is no position left to close)."""
    path = _commands_path(settings, mode)
    try:
        pending = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        pending.append({"action": "close_position", "symbol": symbol})
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(pending), encoding="utf-8")
        tmp.replace(path)
    except (OSError, ValueError):
        pass


def pop_commands(settings: Settings, mode: str) -> list[dict]:
    """Every queued command, oldest first, removed from the queue; commands older than COMMAND_MAX_AGE_SECONDS are dropped."""
    out: list[dict] = []
    legacy = _commands_path(settings, mode)
    if legacy.exists():
        try:
            out += json.loads(legacy.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        try:
            legacy.unlink()
        except OSError:
            pass
    folder = _commands_dir(settings, mode)
    if folder.exists():
        for f in sorted(folder.glob("*.json")):
            try:
                out.append(json.loads(f.read_text(encoding="utf-8")))
                f.unlink()
            except (OSError, ValueError):
                continue
    fresh, seen = [], set()
    for cmd in out:
        key = (cmd.get("action"), cmd.get("symbol"))
        if key in seen:
            continue                      # the same force-exit queued twice (the legacy copy, or a double click): once is enough
        seen.add(key)
        try:
            age = (datetime.now() - datetime.fromisoformat(cmd["queued_at"])).total_seconds() if cmd.get("queued_at") else 0.0
        except ValueError:
            age = 0.0
        if age <= COMMAND_MAX_AGE_SECONDS:
            fresh.append(cmd)
    return fresh


def has_work(settings: Settings, mode: str, cli_symbols: str | None = None) -> bool:
    """A run always has work: either the stock list, or - when there is none - today's trending stocks picked by
    the system itself (auto mode). Open positions to manage count too."""
    return True


def _auto_start_marker(settings: Settings) -> Path:
    return settings.paths.live_data / "state" / "auto_start_paper.json"


def maybe_auto_start_paper(settings: Settings) -> None:
    """Called repeatedly (see auto_start_loop) while the dashboard is running: starts PAPER trading once
    per trading day, as soon as the market opens - so it no longer depends on someone remembering to
    click Start by 9:15. Deliberately PAPER only - LIVE keeps requiring its explicit confirmation phrase
    and capital check, never auto-started, real money is never this automatic."""
    from backend.analytics.dashboard_data import market_status   # local import - avoids a server->analytics
                                                                 # import cycle at module load time
    try:
        ms = market_status(settings)
    except Exception:  # noqa: BLE001 - a transient data/calendar hiccup must not kill the polling loop
        return
    # `trading.auto_start_time` (09:15): the history was loaded after yesterday's close, so the process is ready as soon as it
    # has logged in. No auto-start once the day's square-off time has passed.
    if ms.get("state") not in ("PRE_OPEN", "OPEN", "SQUARE_OFF"):
        return
    cfg = settings.section("trading")
    clock = datetime.fromisoformat(ms["ts"]).time()
    start_at = _hhmm(cfg.get("auto_start_time", "09:00"))
    if clock < start_at or clock >= _hhmm(cfg.get("square_off", "15:00")):
        return
    if not has_work(settings, "paper"):
        return   # nothing to check yet - retried on the next poll once stocks are added in the dashboard
    today = ms["ts"][:10]
    marker = _auto_start_marker(settings)
    if marker.exists():
        try:
            if json.loads(marker.read_text(encoding="utf-8")).get("date") == today:
                return   # already handled today - don't fight a deliberate manual stop
        except (json.JSONDecodeError, OSError):
            pass
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"date": today}), encoding="utf-8")
    if not status(settings, "paper").get("running"):
        start(settings, "paper")


def _hhmm(text: str):
    from datetime import time as _time
    h, m = str(text).split(":")
    return _time(int(h), int(m))


def _forget_stale_auto_start(settings: Settings) -> None:
    """When the dashboard (re)starts: if today's auto-start already happened but that trading process is gone
    WITHOUT a deliberate Stop (it crashed, the PC rebooted), forget the marker so this launch starts it again.
    A deliberate Stop writes `stopped_at` to the state file and is left alone."""
    marker = _auto_start_marker(settings)
    try:
        today = datetime.now().date().isoformat()
        if not marker.exists() or json.loads(marker.read_text(encoding="utf-8")).get("date") != today:
            return
        info = status(settings, "paper")
        if not info.get("running") and "stopped_at" not in info:
            marker.unlink()
    except (OSError, json.JSONDecodeError):
        pass


def auto_start_loop(settings: Settings, stop_event, poll_seconds: float = 15.0) -> None:
    """Run in a daemon thread for the dashboard process's whole lifetime (see scripts/dashboard.py)."""
    _forget_stale_auto_start(settings)
    while not stop_event.is_set():
        try:
            maybe_auto_start_paper(settings)
        except Exception:  # noqa: BLE001 - this loop must never take the dashboard down
            pass
        stop_event.wait(poll_seconds)
