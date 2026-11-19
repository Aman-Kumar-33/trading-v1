"""Settings loader.

Reads config/settings.yaml once and exposes it as a plain dict plus resolved filesystem paths.
Relative paths in the YAML are resolved against the algo-system/ root, so scripts work from any cwd.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

if getattr(sys, "frozen", False):
    # Packaged as Dashboard.exe (see scripts/build_dashboard_exe.py): config/, data/, logs/ and
    # backtest_data/ are real, user-editable directories next to the exe, not bundled into it - only the
    # frontend's static HTML/JS/CSS are bundled (backend/api/dashboard_server.py handles that separately).
    ROOT = Path(sys.executable).resolve().parent
    # both_systems.exe may sit one folder above the project (D:\Business\algo trading\v17 anthropic\): then the project is the
    # `second-system` folder next to it. SECOND_SYSTEM_ROOT overrides both.
    if not (ROOT / "config" / "settings.yaml").exists():
        for _cand in (ROOT / "second-system", ROOT.parent / "second-system"):
            if (_cand / "config" / "settings.yaml").exists():
                ROOT = _cand
                break
    import os as _os
    if _os.environ.get("SECOND_SYSTEM_ROOT"):
        ROOT = Path(_os.environ["SECOND_SYSTEM_ROOT"]).resolve()
else:
    ROOT = Path(__file__).resolve().parents[2]  # .../algo-system
DEFAULT_SETTINGS_FILE = ROOT / "config" / "settings.yaml"


@dataclass(frozen=True)
class Paths:
    root: Path
    backtest_data: Path
    live_data: Path
    logs: Path

    # backtest_data layout
    @property
    def raw(self) -> Path:
        return self.backtest_data / "raw" / "yfinance"

    @property
    def clean(self) -> Path:
        return self.backtest_data / "clean"

    @property
    def corporate_actions(self) -> Path:
        return self.backtest_data / "corporate_actions"

    @property
    def universe(self) -> Path:
        return self.backtest_data / "universe"

    @property
    def quality_reports(self) -> Path:
        return self.backtest_data / "quality_reports"

    @property
    def manifest(self) -> Path:
        return self.backtest_data / "manifest"

    @property
    def results(self) -> Path:
        return self.backtest_data / "results"

    # live data layout
    @property
    def db_file(self) -> Path:
        return self.live_data / "algo.db"

    def ensure(self) -> None:
        for p in (self.raw, self.clean, self.corporate_actions, self.universe,
                  self.quality_reports, self.manifest, self.results, self.live_data, self.logs):
            p.mkdir(parents=True, exist_ok=True)


class Settings(dict):
    """The parsed YAML as a dict, plus `.paths` and a dotted-key `get_path` helper."""

    def __init__(self, data: dict[str, Any], root: Path = ROOT):
        super().__init__(data)
        p = data.get("paths", {})
        self.paths = Paths(
            root=root,
            backtest_data=_resolve(root, p.get("backtest_data", "../backtest_data")),
            live_data=_resolve(root, p.get("live_data", "data")),
            logs=_resolve(root, p.get("logs", "logs")),
        )

    def section(self, name: str) -> dict[str, Any]:
        return self.get(name, {}) or {}

    def resolve(self, relative: str) -> Path:
        return _resolve(self.paths.root, relative)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def settings_file() -> Path:
    """config/settings.yaml, or the file named by the ALGO_SETTINGS_FILE environment variable (a bare name is looked
    up in config/). The 5 minute instance sets it, and its trading process inherits it, so both read the same file."""
    name = os.environ.get("ALGO_SETTINGS_FILE", "").strip()
    if not name:
        return DEFAULT_SETTINGS_FILE
    p = Path(name)
    return p if p.is_absolute() else ROOT / "config" / p


def _read_yaml(path: Path) -> dict[str, Any]:
    """A settings file; `extends: other.yaml` loads that file (next to this one) first and merges this one over it."""
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    base = data.pop("extends", None)
    return _deep_merge(_read_yaml(path.parent / base), data) if base else data


def load_settings(path: Path | str | None = None, overrides: dict[str, Any] | None = None) -> Settings:
    data = _read_yaml(Path(path) if path else settings_file())
    if overrides:
        data = _deep_merge(data, overrides)
    return Settings(data)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


def _deep_merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
