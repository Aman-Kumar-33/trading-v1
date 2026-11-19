"""Nothing the dashboards start may open a terminal window: no console programs for status checks, hidden consoles for children.

Run from second-system/:  python -m pytest tests -q
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.execution import process_control as pc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_a_status_check_starts_no_program(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a process-alive check must not start another program")
    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(subprocess, "Popen", boom)
    assert pc._is_running(os.getpid()) is True
    assert pc._is_running(0) is False and pc._is_running(-5) is False


def test_a_finished_process_is_not_running():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    assert pc._is_running(p.pid) is False
    q = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert pc._is_running(q.pid) is True
    finally:
        q.kill()
        q.wait()
    assert pc._is_running(q.pid) is False


@pytest.mark.skipif(sys.platform != "win32", reason="console windows are a Windows thing")
def test_the_launchers_give_children_a_hidden_console_not_none():
    for rel in ("scripts/both_systems.py", "backend/api/dashboard_server.py"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "CREATE_NO_WINDOW" in text and "DETACHED_PROCESS" not in text.replace("NOT \"no console\" (DETACHED_PROCESS)", "")
    assert "tasklist" not in (ROOT / "backend/execution/process_control.py").read_text(encoding="utf-8").replace("`tasklist`", "")
