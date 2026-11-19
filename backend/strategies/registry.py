"""Strategy configs (versioned YAML parameters) and the strategy vault (immutable version snapshots).

Layout:
    backend/strategies/configs/<strategy_id>.yaml   - {"v1": {...params...}, "v2": {...}, ...}
    strategy_vault/<strategy_id>/<version>/         - meta.json + params.yaml, written once, never edited

The evolution loop (Phase 7) creates new config versions and new vault snapshots; it never edits an
existing version's files or the live version's config in place - that's what makes rollback trivial
(point back at an older version's snapshot).
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from backend.strategies.base import Strategy

CONFIGS_DIR = Path(__file__).resolve().parent / "configs"


# ---------------------------------------------------------------------- config file I/O
def load_params(strategy_id: str, version: str = "v1", configs_dir: Path = CONFIGS_DIR) -> dict[str, Any]:
    path = configs_dir / f"{strategy_id}.yaml"
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return dict(data.get(version, {}))


def list_versions(strategy_id: str, configs_dir: Path = CONFIGS_DIR) -> list[str]:
    path = configs_dir / f"{strategy_id}.yaml"
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return sorted(data.keys())


def save_params(strategy_id: str, version: str, params: dict[str, Any], configs_dir: Path = CONFIGS_DIR) -> Path:
    """Add (or, for an already-existing version, overwrite) one version's params in the strategy's YAML
    file. In normal operation new tuning always goes into a *new* version name - the evolution loop
    should never call this with a version that's already been promoted/traded."""
    configs_dir.mkdir(parents=True, exist_ok=True)
    path = configs_dir / f"{strategy_id}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    data = data or {}
    data[version] = params
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def build_strategy(cls: type[Strategy], version: str = "v1", configs_dir: Path = CONFIGS_DIR,
                   overrides: dict[str, Any] | None = None) -> Strategy:
    """Instantiate a strategy with its versioned config, applying any ad-hoc overrides on top (e.g. for
    a parameter sweep in Phase 3 - the sweep never touches the saved YAML)."""
    params = {**load_params(cls.strategy_id, version, configs_dir), **(overrides or {})}
    return cls(params=params, version=version)


# ---------------------------------------------------------------------- strategy vault (immutable snapshots)
def snapshot(strategy: Strategy, vault_root: Path, status: str = "RESEARCH", notes: str = "",
            parent_version: str | None = None) -> Path:
    """Write an immutable snapshot of this strategy version to strategy_vault/. Refuses to overwrite an
    existing snapshot - once written, a version's record never changes (that's the whole point)."""
    out_dir = vault_root / strategy.strategy_id / strategy.version
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} already exists - versions are immutable, use a new version name")
    out_dir.mkdir(parents=True)
    (out_dir / "params.yaml").write_text(yaml.safe_dump(strategy.params, sort_keys=False), encoding="utf-8")
    meta = {
        "strategy_id": strategy.strategy_id, "version": strategy.version, "horizon": strategy.horizon.value,
        "status": status, "notes": notes, "parent_version": parent_version, "code_hash": strategy.config_hash(),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir


def load_snapshot(vault_root: Path, strategy_id: str, version: str) -> dict[str, Any]:
    out_dir = vault_root / strategy_id / version
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    meta["params"] = yaml.safe_load((out_dir / "params.yaml").read_text(encoding="utf-8")) or {}
    return meta


def list_snapshots(vault_root: Path, strategy_id: str) -> list[str]:
    d = vault_root / strategy_id
    return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.exists() else []


# ---------------------------------------------------------------------- DB registration (strategy_versions table)
def register_version(conn: sqlite3.Connection, strategy: Strategy, status: str = "RESEARCH",
                     parent_version: str | None = None, notes: str = "") -> None:
    conn.execute(
        """INSERT INTO strategy_versions (strategy_id, version, created_at, status, params_json,
                                          code_hash, parent_version, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(strategy_id, version) DO UPDATE SET status=excluded.status, notes=excluded.notes""",
        (strategy.strategy_id, strategy.version, datetime.now(timezone.utc).isoformat(timespec="seconds"),
         status, json.dumps(strategy.params, default=str), strategy.config_hash(), parent_version, notes))


def get_status(conn: sqlite3.Connection, strategy_id: str, version: str) -> str | None:
    row = conn.execute("SELECT status FROM strategy_versions WHERE strategy_id=? AND version=?",
                       (strategy_id, version)).fetchone()
    return row["status"] if row else None
