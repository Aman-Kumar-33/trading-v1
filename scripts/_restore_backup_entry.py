"""Entry point bundled into Backup.exe - extracts the snapshot it carries into ./algo-system-restored
next to wherever the exe is run from."""
import sys
import zipfile
from pathlib import Path


def main() -> int:
    frozen = getattr(sys, "frozen", False)
    bundle_dir = Path(sys._MEIPASS) if frozen else Path(__file__).resolve().parent.parent
    exe_dir = Path(sys.executable).resolve().parent if frozen else Path(__file__).resolve().parent.parent
    dest = exe_dir / "algo-system-restored"
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_dir / "snapshot.zip") as zf:
        zf.extractall(dest)
    print(f"Restored to: {dest}")
    input("Press Enter to close...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
