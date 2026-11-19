"""Run once after you've sent your bot any message on Telegram: prints the chat id(s) found so you can
paste one into TELEGRAM_CHAT_ID in .env. Reads TELEGRAM_BOT_TOKEN from .env via the same loader the rest
of the system uses - no new dependency, one GET request to Telegram's own getUpdates endpoint.

    python scripts/get_telegram_chat_id.py
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.core.secrets import load_env  # noqa: E402


def main() -> int:
    load_env()
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print("TELEGRAM_BOT_TOKEN is not set in .env - add it first.")
        return 1
    with urllib.request.urlopen(f"https://api.telegram.org/bot{token}/getUpdates", timeout=10) as resp:
        data = json.loads(resp.read())
    chats = {u["message"]["chat"]["id"]: u["message"]["chat"].get("username") or u["message"]["chat"].get("first_name")
            for u in data.get("result", []) if "message" in u}
    if not chats:
        print("No messages found yet - open Telegram, message your bot anything, then re-run this.")
        return 1
    for chat_id, name in chats.items():
        print(f"chat_id={chat_id}  ({name})")
    print("\nPaste one of the chat_id values above into TELEGRAM_CHAT_ID in .env")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
