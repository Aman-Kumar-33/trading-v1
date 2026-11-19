"""Structured logging.

Every log record goes to two places:
  * console - short human-readable lines
  * logs/<name>.jsonl - one JSON object per line, machine-readable

The JSON logs are the raw material for the weekly evolution review, so attach structured context with
`log.info("message", extra={"data": {...}})` instead of formatting values into the message string.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from backend.core.clock import IST

_CONFIGURED: set[str] = set()


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            # IST, not UTC - every other timestamp in this system (session state, market calendar, candle
            # ts) is IST; logging in UTC made every log line look ~5:30 off from the market clock (an
            # entry at 9:45am IST logged as "04:15") while console output (local machine tz) looked right.
            "ts": datetime.fromtimestamp(record.created, tz=IST).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        data = getattr(record, "data", None)
        if data is not None:
            payload["data"] = data
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def setup_logging(name: str, log_dir: Path, level: int = logging.INFO) -> logging.Logger:
    """Configure the root logger once per log file name and return a logger called `name`."""
    root = logging.getLogger()
    if name not in _CONFIGURED:
        log_dir.mkdir(parents=True, exist_ok=True)
        root.setLevel(level)

        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"))
        console.setLevel(level)

        jsonl = logging.FileHandler(log_dir / f"{name}.jsonl", encoding="utf-8")
        jsonl.setFormatter(JsonFormatter())
        jsonl.setLevel(logging.DEBUG)

        root.addHandler(console)
        root.addHandler(jsonl)
        # yfinance is chatty at INFO/DEBUG; we log its errors ourselves.
        logging.getLogger("yfinance").setLevel(logging.CRITICAL)
        _CONFIGURED.add(name)
    return logging.getLogger(name)
