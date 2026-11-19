"""Broker credentials from `.env` (never hardcoded, never committed - see `.gitignore`).

`load_dotenv()` is safe to call even when `.env` doesn't exist yet (a no-op), so every entry point can
call `load_env()` unconditionally.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from backend.core.config import ROOT

_loaded = False


def load_env() -> None:
    global _loaded
    if not _loaded:
        load_dotenv(ROOT / ".env")
        _loaded = True


@dataclass(frozen=True)
class AngelCredentials:
    api_key: str
    client_code: str
    pin: str
    totp_secret: str


class MissingCredentials(RuntimeError):
    pass


def get_angel_credentials() -> AngelCredentials:
    load_env()
    values = {name: os.environ.get(name, "").strip() for name in
             ("ANGEL_API_KEY", "ANGEL_CLIENT_CODE", "ANGEL_PIN", "ANGEL_TOTP_SECRET")}
    missing = [k for k, v in values.items() if not v]
    if missing:
        raise MissingCredentials(
            f"missing {', '.join(missing)} - copy .env.example to .env and fill them in "
            f"(see the plan's 'Angel One SmartAPI setup' section for how to get them)")
    return AngelCredentials(values["ANGEL_API_KEY"], values["ANGEL_CLIENT_CODE"], values["ANGEL_PIN"],
                            values["ANGEL_TOTP_SECRET"])
