"""Telegram alerts on every entry/exit. Reads TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID from the environment
(.env) - a no-op (logged once) if either is unset, never raises into the caller.

Sending happens on a background thread: the trading loop is single-threaded asyncio, and a slow Telegram
round-trip (two real timeouts were logged on 2026-11-11) used to freeze tick handling for up to 5 seconds.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import urllib.parse
import urllib.request

log = logging.getLogger(__name__)
_warned = False
_label = ""   # e.g. "5M": a second system's messages are tagged so they can't be mistaken for the main system's
ATTEMPTS = 3
TIMEOUT_SECONDS = 10


def _credentials() -> tuple[str | None, str | None]:
    # .env is loaded lazily by backend.core.secrets; call it here so a script that never logs in to the
    # broker (the dashboard, a test) still finds the Telegram settings.
    from backend.core.secrets import load_env
    load_env()
    return os.environ.get("TELEGRAM_BOT_TOKEN", "").strip() or None, os.environ.get("TELEGRAM_CHAT_ID", "").strip() or None


def set_label(label: str) -> None:
    global _label
    _label = (label or "").strip()


def send_sync(text: str) -> tuple[bool, str]:
    """Send one message, retrying on transient errors. Returns (ok, detail) - used by the connection test."""
    token, chat_id = _credentials()
    if _label:
        text = f"[{_label}] {text}"
    if not token or not chat_id:
        return False, "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set in .env"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
    detail = ""
    for attempt in range(1, ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=TIMEOUT_SECONDS) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            if body.get("ok"):
                return True, "sent"
            detail = f"Telegram rejected the message: {body.get('description')}"
            break   # a rejection (bad token / chat id) won't fix itself on retry
        except Exception as exc:  # noqa: BLE001 - network hiccup; try again
            detail = repr(exc)
    return False, detail


def notify(text: str) -> None:
    """Fire-and-forget: returns immediately, the send (with retries) runs on a daemon thread."""
    global _warned
    token, chat_id = _credentials()
    if not token or not chat_id:
        if not _warned:
            log.warning("Telegram not configured (TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID unset) - alerts disabled")
            _warned = True
        return

    def _run() -> None:
        ok, detail = send_sync(text)
        if not ok:
            log.error("telegram notify failed", extra={"data": {"error": detail}})

    threading.Thread(target=_run, name="telegram-notify", daemon=True).start()


def _money(x: float) -> str:
    return f"Rs {x:,.2f}"


def format_entry(symbol: str, strategy: str, side: str, qty: int, price: float, mode: str, when: str) -> str:
    return (f"{'BUY' if side == 'BUY' else 'SELL'} - {mode}\n"
            f"Stock: {symbol}\n"
            f"Strategy: {strategy}\n"
            f"Side: {side}\n"
            f"Quantity: {qty}\n"
            f"Price: {_money(price)}\n"
            f"Total {'buy' if side == 'BUY' else 'sell'} value: {_money(qty * price)}\n"
            f"P&L: - (position opened)\n"
            f"Time: {when}")


def format_exit(symbol: str, strategy: str, side: str, qty: int, price: float, entry_price: float,
                pnl_gross: float, costs: float, pnl_net: float, reason: str, mode: str, when: str) -> str:
    exit_side = "SELL" if side == "BUY" else "BUY"
    return (f"{exit_side} - {mode}\n"
            f"Stock: {symbol}\n"
            f"Strategy: {strategy}\n"
            f"Side: {exit_side} (closing {side})\n"
            f"Quantity: {qty}\n"
            f"Price: {_money(price)}  (entry {_money(entry_price)})\n"
            f"Total {'sell' if exit_side == 'SELL' else 'buy'} value: {_money(qty * price)}\n"
            f"P&L: {pnl_net:+,.2f} net  ({pnl_gross:+,.2f} gross - {costs:,.2f} charges)\n"
            f"Exit reason: {reason}\n"
            f"Time: {when}")


def format_startup(what: str, mode: str, timeframe: str, strategy: str, capital: float, min_pct: float,
                   max_pct: float, max_open: int, stocks: list[str], source: str, when: str) -> str:
    listing = ", ".join(stocks) if stocks else "none yet - the list is made after the close for the next session (if missing: picked before the open); trading from 09:15"
    return (f"{what} STARTED - {mode}\n"
            f"Strategy: {strategy} ({timeframe})\n"
            f"Capital: {_money(capital)}\n"
            f"Trade size: {min_pct:.0%} to {max_pct:.0%} of capital (system decides), "
            f"{f'max {max_open} open positions' if max_open else 'no limit on open positions (free cash decides)'}\n"
            f"Stocks ({source}): {listing}\n"
            f"Time: {when}")


def format_picks(mode: str, timeframe: str, picks: list[dict], when: str) -> str:
    lines = [f"{p['rank']}. {p['symbol']}  score {p['score']:.0f}  {p.get('crosses_per_session', 0):.1f} crosses/day  "
             f"MACD {p.get('net_return_pct', 0):+.1f}%  today {p.get('pct_change', 0):+.1f}%" for p in picks]
    return (f"TRENDING STOCKS PICKED - {mode}\n"
            f"Only these will be bought today ({timeframe} MACD):\n" + "\n".join(lines) + f"\nTime: {when}")


def format_signal(symbol: str, strategy: str, side: str, timeframe: str, bar_start: str, close: float,
                  reason: str, action: str, mode: str, live: bool = False) -> str:
    """One MACD signal, whether or not it led to a trade - `action` says what the system did with it."""
    cross = "bullish cross (MACD above signal)" if side == "BUY" else "bearish cross (MACD below signal)"
    return (f"{side} SIGNAL - {mode}\n"
            f"Stock: {symbol}\n"
            f"Strategy: {strategy} ({timeframe})\n"
            f"Signal: MACD {cross}\n"
            + (f"Time: {bar_start} IST, live price {_money(close)}\n" if live else
               f"Bar: {bar_start} IST, closed at {_money(close)}\n") +
            f"Detail: {reason}\n"
            f"Action: {action}")
