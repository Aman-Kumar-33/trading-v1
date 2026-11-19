"""Builds Backup.exe - a self-contained snapshot of this entire project folder (code, config, scripts,
tests, frontend source, data/, logs/, backups/), so a disk failure or bad edit can be undone by running
it. Excludes only what's regenerable, not part of "the system": .venv/, node_modules/, dist/, build/,
__pycache__/, .git/ (reinstalled via requirements.txt/npm install, not something to restore).

    python scripts/build_backup_exe.py

Running the resulting Backup.exe anywhere extracts the snapshot into ./algo-system-restored next to it.
"""
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXCLUDE_DIRS = {".venv", "node_modules", "dist", "build", "__pycache__", ".git"}
EXCLUDE_SUFFIXES = {".pyc"}


def make_snapshot(zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in ROOT.rglob("*"):
            if p.is_dir() or p.suffix in EXCLUDE_SUFFIXES or any(part in EXCLUDE_DIRS for part in p.parts):
                continue
            zf.write(p, p.relative_to(ROOT))


def main() -> int:
    snapshot = ROOT / "build" / "snapshot.zip"
    make_snapshot(snapshot)
    print(f"  snapshot: {snapshot} ({snapshot.stat().st_size / 1e6:.1f} MB)")

    sep = ";" if sys.platform == "win32" else ":"
    args = [sys.executable, "-m", "PyInstaller", "--onefile", "--name", "Backup", "--console",
           "--add-data", f"{snapshot}{sep}.",
           "--distpath", str(ROOT / "dist"), "--workpath", str(ROOT / "build"), "--specpath", str(ROOT / "build"),
           str(ROOT / "scripts" / "_restore_backup_entry.py")]
    result = subprocess.run(args, cwd=str(ROOT))
    if result.returncode == 0:
        print(f"\n  built {ROOT / 'dist' / 'Backup.exe'}")
        print("  copy it anywhere and double-click to restore a full source snapshot next to it")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
