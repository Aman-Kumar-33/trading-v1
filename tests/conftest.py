"""Tests must never talk to the real Telegram bot: OrderManager/SimpleRunner send alerts on every simulated
fill, and backend.notify.telegram loads TELEGRAM_* from the real .env. Blank values already in the environment
win over .env (python-dotenv does not override), so the sender sees "not configured" and does nothing."""
import os

os.environ["TELEGRAM_BOT_TOKEN"] = ""
os.environ["TELEGRAM_CHAT_ID"] = ""
