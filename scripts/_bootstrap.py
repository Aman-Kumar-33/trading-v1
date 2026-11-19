"""Shared setup for command-line scripts: import path, settings, logging."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.core.config import get_settings  # noqa: E402
from backend.core.logging_setup import setup_logging  # noqa: E402


def init(name: str):
    settings = get_settings()
    settings.paths.ensure()
    log = setup_logging(name, settings.paths.logs)
    from backend.notify import telegram as tg
    tg.set_label(settings.section("instance").get("label", ""))   # the 5 minute system tags its Telegram messages
    return settings, log
