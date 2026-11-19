"""Build Dashboard.exe (PyInstaller, one file). Run once per release - not part of the normal test suite,
and not needed to use the dashboard (`python scripts/dashboard.py` works without building anything).

    python scripts/build_dashboard_exe.py

Produces dist/Dashboard.exe. Copy or run it from inside algo-system/ (next to config/, data/,
backtest_data/, scripts/) - it resolves those the same way every other script does, relative to where
it's running from (see backend/core/config.py's frozen-exe branch). The React frontend is built first
(npm ci + vite build) and its dist/ output is bundled into the exe as read-only static assets
(backend/api/dashboard_server.py's frozen-exe branch) - re-extracting them on every run loses nothing.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend" / "dashboard-react"


def main() -> int:
    npm = "npm.cmd" if sys.platform == "win32" else "npm"
    print("  building React frontend...")
    install = subprocess.run([npm, "ci"], cwd=str(FRONTEND), shell=(sys.platform == "win32"))
    if install.returncode != 0:
        return install.returncode
    build = subprocess.run([npm, "run", "build"], cwd=str(FRONTEND), shell=(sys.platform == "win32"))
    if build.returncode != 0:
        return build.returncode

    sep = ";" if sys.platform == "win32" else ":"
    args = [
        sys.executable, "-m", "PyInstaller", "--onefile", "--name", "second_dashboard", "--console",
        "--add-data", f"{FRONTEND / 'dist'}{sep}frontend/dashboard-react/dist",
        "--distpath", str(ROOT / "dist"), "--workpath", str(ROOT / "build"),
        "--specpath", str(ROOT / "build"),
        str(ROOT / "scripts" / "dashboard.py"),
    ]
    print("  running:", " ".join(args))
    result = subprocess.run(args, cwd=str(ROOT))
    if result.returncode == 0:
        print(f"\n  built {ROOT / 'dist' / 'second_dashboard.exe'}")
        print("  copy it into algo-system/ (next to config/, scripts/, data/) and double-click to run")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
