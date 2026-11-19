"""Minimal dashboard HTTP server - stdlib only, no new runtime dependency, since this only needs to serve
a handful of small JSON endpoints plus three static files to one local browser tab. Binds to 127.0.0.1 -
not meant for concurrent users or the open internet.

Routes:
  GET  /, /*                                 -> the static React build (frontend/dashboard-react/dist)
  GET  /api/overview                         -> capital + paper/live aggregate + TODAY-only metrics
  GET  /api/daily_pnl?mode=paper|live&days=30 -> net P&L grouped by calendar day
  GET  /api/strategies                       -> per-strategy backtest/paper/live metrics + status
  GET  /api/ensemble?mode=paper|live         -> saved ensemble weights/history
  GET  /api/positions?mode=paper|live        -> open positions + live ltp/unrealized_pnl
  GET  /api/status                           -> {paper: {...}, live: {...}, capital: {...}}
  GET  /api/watchlist?mode=paper|live        -> the stocks the MACD system is allowed to check
  POST /api/watchlist {mode, symbols}        -> replace that list (typos are reported back, not saved)
  GET  /api/trending?mode=paper|live         -> today's auto-picked trending stocks + which list is in use
  GET  /api/market_status                    -> {state, is_open, ts}
  GET  /api/trades?mode=paper|live&limit=500 -> raw closed-trade rows from the DB
  GET  /api/backtests                        -> list of available backtest runs (picker metadata)
  GET  /api/backtests/<kind>/<id>            -> full run detail: summary, metrics, equity curve, folds
  GET  /api/strategy_equity?id=&mode=paper|live -> that strategy's live/paper equity curve from the DB
  GET  /api/candles?symbol=&timeframe=30m&limit=300&mode= -> OHLC candles + MACD lines + MACD cross signals + trade markers
  GET  /api/events?mode=paper|live&since=    -> parsed log lines newer than `since` (drives animation)
  GET  /api/logs?source=run_trading_paper&lines=200
  POST /api/paper/start {symbols?}  | /api/paper/stop
  POST /api/live/start {confirm: "I_ACCEPT_REAL_MONEY_RISK", symbols?}  | /api/live/stop
"""
from __future__ import annotations

import json
import mimetypes
import os
import sqlite3
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from backend.analytics import backtest_views as bv
from backend.analytics import dashboard_data as dd
from backend.core.config import Settings
from backend.data import watchlist
from backend.execution import process_control as pc

if getattr(sys, "frozen", False):
    # PyInstaller onefile: the frontend's static files are bundled and extracted to this temp dir -
    # read-only assets, so re-extracting on every run (losing nothing) is the right tradeoff, unlike
    # config/data/logs which stay next to the exe (see backend/core/config.py).
    FRONTEND_DIR = Path(sys._MEIPASS) / "frontend" / "dashboard-react" / "dist"
else:
    FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend" / "dashboard-react" / "dist"
LIVE_CONFIRM_PHRASE = "I_ACCEPT_REAL_MONEY_RISK"


def _send_json(handler: BaseHTTPRequestHandler, payload, status: int = 200) -> None:
    body = json.dumps(payload, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _send_static(handler: BaseHTTPRequestHandler, path: Path) -> None:
    if not path.is_file():
        return _send_json(handler, {"error": "not found"}, 404)
    body = path.read_bytes()
    handler.send_response(200)
    handler.send_header("Content-Type", mimetypes.guess_type(str(path))[0] or "application/octet-stream")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _resolve_static(url_path: str) -> Path | None:
    """Any file under the built React app, traversal-guarded - needed because Vite's build emits hashed
    asset filenames under assets/, unlike the old 3-file hand-written frontend. Falls back to index.html
    for any unknown path (no router yet, but safe for a future client-side one)."""
    rel = url_path.lstrip("/") or "index.html"
    candidate = (FRONTEND_DIR / rel).resolve()
    try:
        candidate.relative_to(FRONTEND_DIR.resolve())
    except ValueError:
        return None
    if not candidate.is_file():
        candidate = FRONTEND_DIR / "index.html"
    return candidate


def make_handler(settings: Settings) -> type[BaseHTTPRequestHandler]:
    db_lock = threading.Lock()

    def with_db(fn):
        conn = sqlite3.connect(str(settings.paths.db_file))
        conn.row_factory = sqlite3.Row
        try:
            with db_lock:
                return fn(conn)
        finally:
            conn.close()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass   # quiet - the dashboard polls every few seconds, don't spam stdout

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path, qs = parsed.path, parse_qs(parsed.query)
            if path.startswith("/open/"):
                return _open_instance(self, settings, unquote(path[len("/open/"):]))
            if not path.startswith("/api/"):
                target = _resolve_static(path)
                return _send_json(self, {"error": "forbidden"}, 403) if target is None else _send_static(self, target)
            try:
                if path == "/api/overview":
                    return _send_json(self, with_db(lambda conn: dd.overview(settings, conn)))
                if path == "/api/strategies":
                    return _send_json(self, with_db(lambda conn: dd.strategy_rows(settings, conn)))
                if path == "/api/ensemble":
                    return _send_json(self, dd.ensemble_weights(settings, qs.get("mode", ["paper"])[0]))
                if path == "/api/positions":
                    mode = qs.get("mode", ["paper"])[0]
                    return _send_json(self, with_db(lambda conn: dd.positions_with_pnl(settings, conn, mode)))
                if path == "/api/status":
                    return _send_json(self, {"paper": pc.status(settings, "paper"), "live": pc.status(settings, "live"),
                                             "capital": dd.load_capital(settings), "config": dd.trading_config(settings)})
                if path == "/api/watchlist":
                    mode = qs.get("mode", ["paper"])[0]
                    return _send_json(self, {"symbols": watchlist.load(settings, mode) or [],
                                             "updated_at": watchlist.updated_at(settings, mode)})
                if path == "/api/trending":
                    mode = qs.get("mode", ["paper"])[0]
                    return _send_json(self, dd.trending_view(settings, mode))
                if path == "/api/market_status":
                    return _send_json(self, dd.market_status(settings))
                if path == "/api/trades":
                    mode = qs.get("mode", ["paper"])[0]
                    limit = int(qs.get("limit", ["500"])[0])
                    return _send_json(self, with_db(lambda conn: dd.db_trade_rows(conn, mode, limit)))
                if path == "/api/backtests":
                    return _send_json(self, bv.list_backtest_runs(settings))
                if path.startswith("/api/backtests/"):
                    parts = [p for p in path.split("/")[3:] if p]
                    if len(parts) != 2:
                        return _send_json(self, {"error": "not found"}, 404)
                    kind, run_id = parts
                    try:
                        return _send_json(self, bv.backtest_run_detail(settings, kind, run_id))
                    except FileNotFoundError as exc:
                        return _send_json(self, {"error": str(exc)}, 404)
                    except ValueError as exc:
                        return _send_json(self, {"error": str(exc)}, 400)
                if path == "/api/daily_pnl":
                    mode = qs.get("mode", ["paper"])[0]
                    days = int(qs.get("days", ["30"])[0])
                    return _send_json(self, with_db(lambda conn: dd.daily_pnl(conn, mode, days)))
                if path == "/api/candles":
                    symbol = qs.get("symbol", [""])[0]
                    timeframe = qs.get("timeframe", ["5m"])[0]
                    limit = int(qs.get("limit", ["300"])[0])
                    mode = qs.get("mode", ["paper"])[0]
                    if not symbol:
                        return _send_json(self, {"error": "symbol is required"}, 400)
                    # load extra bars before the visible window: MACD is an EMA of EMAs, so its first visible
                    # values are only right if it has run over the bars before them
                    warmup = 400
                    candles_df = with_db(lambda conn: dd.symbol_candles(settings, conn, symbol, timeframe, limit + warmup))
                    tcfg = settings.section("trading")
                    confirm = ((float(tcfg.get("macd_cross_tolerance", 0.0)), float(tcfg.get("macd_cross_tolerance_atr", 0.0)))
                               if tcfg.get("signal_mode") == "candle_confirmed" else None)
                    from backend.core.timeframes import minutes as tf_minutes
                    sq = None if tcfg.get("carry_longs", False) else tcfg.get("square_off")
                    try:
                        tfm = tf_minutes(timeframe)
                    except ValueError:                     # 1d
                        tfm = None
                    shown = dd.macd_overlay(candles_df, limit, confirm, square_off=sq, tf_minutes=tfm)
                    trades = with_db(lambda conn: dd.db_trade_rows(conn, mode, 100, symbol))
                    return _send_json(self, {"candles": shown["candles"].to_dict("records"), "macd": shown["macd"],
                                             "signals": shown["signals"], "trades": trades,
                                             "stale": bool(candles_df.attrs.get("stale", False)),
                                             "context": dd.chart_context(settings, mode, symbol)})
                if path == "/api/strategy_equity":
                    sid = qs.get("id", [""])[0]
                    mode = qs.get("mode", ["paper"])[0]
                    return _send_json(self, with_db(lambda conn: bv.live_strategy_equity(settings, conn, sid, mode)))
                if path == "/api/events":
                    mode = qs.get("mode", ["paper"])[0]
                    since = qs.get("since", [None])[0]
                    return _send_json(self, dd.pipeline_events(settings, f"run_trading_{mode}", since))
                if path == "/api/logs":
                    source = qs.get("source", ["run_trading_paper"])[0]
                    n = int(qs.get("lines", ["200"])[0])
                    return _send_json(self, dd.tail_log(settings, source, n))
                return _send_json(self, {"error": "not found"}, 404)
            except Exception as exc:  # noqa: BLE001 - one bad request must not take the server down
                return _send_json(self, {"error": repr(exc)}, 500)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                body = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                body = {}
            try:
                if path == "/api/watchlist":
                    if watchlist.follow_settings(settings) is not None:
                        return _send_json(self, {"error": "this system trades the stocks the main system trades - "
                                                          "change the list on the main system's page"}, 400)
                    mode = body.get("mode", "paper")
                    valid, unknown = watchlist.split_known(settings, watchlist.parse(body.get("symbols")))
                    saved = watchlist.save(settings, mode, valid)
                    return _send_json(self, {"symbols": saved, "unknown": unknown,
                                             "updated_at": watchlist.updated_at(settings, mode),
                                             "process_running": bool(pc.status(settings, mode).get("running"))})
                if path == "/api/paper/start":
                    if not pc.has_work(settings, "paper", body.get("symbols")):
                        return _send_json(self, {"error": "add the stocks to check in the Paper Trading box first - "
                                                          "with no stock list there is nothing to run"}, 400)
                    return _send_json(self, pc.start(settings, "paper", body.get("symbols")))
                if path == "/api/paper/stop":
                    return _send_json(self, pc.stop(settings, "paper"))
                if path == "/api/live/start":
                    if settings.section("instance").get("live_enabled", True) is False:
                        return _send_json(self, {"error": "live trading is not available in this system"}, 403)
                    if body.get("confirm") != LIVE_CONFIRM_PHRASE:
                        return _send_json(self, {"error": f"send {{\"confirm\": \"{LIVE_CONFIRM_PHRASE}\"}} to "
                                                          f"start live trading - this is deliberately not a single click"}, 400)
                    if dd.load_capital(settings)["capital"] <= 0:
                        return _send_json(self, {"error": "trading.capital is 0 in config/settings.yaml - "
                                                          "set your real capital first"}, 400)
                    if not pc.has_work(settings, "live", body.get("symbols")):
                        return _send_json(self, {"error": "add the stocks to check in the Live Trading box first"}, 400)
                    return _send_json(self, pc.start(settings, "live", body.get("symbols")))
                if path == "/api/live/stop":
                    return _send_json(self, pc.stop(settings, "live"))
                if path.startswith("/api/positions/") and path.endswith("/close"):
                    symbol = unquote(path[len("/api/positions/"):-len("/close")])
                    mode = body.get("mode", "paper")
                    # queued even when no trading process is running: the next one to start picks it up as its
                    # first action, so a position can be flagged for exit before the market opens
                    pc.queue_close_position(settings, mode, symbol)
                    return _send_json(self, {"queued": True, "symbol": symbol, "mode": mode,
                                             "process_running": bool(pc.status(settings, mode).get("running"))})
                return _send_json(self, {"error": "not found"}, 404)
            except Exception as exc:  # noqa: BLE001
                return _send_json(self, {"error": repr(exc)}, 500)

    return Handler


def _port_open(host: str, port: int) -> bool:
    import socket
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _open_instance(handler, settings, key: str) -> None:
    """The sidebar's "5 Minute System" entry: another dashboard (own port, own data) opened in a new tab. If its
    server is not running yet it is started first (it never starts trading by itself), then the tab is redirected to it."""
    import subprocess
    import sys
    import time
    link = next((l for l in settings.section("instance").get("links", []) if l.get("key") == key), None)
    if link is None:
        return _send_json(handler, {"error": "unknown system"}, 404)
    parsed = urlparse(link["url"])
    host, port = parsed.hostname or "127.0.0.1", parsed.port or 80
    if not _port_open(host, port):
        root = settings.paths.root
        if getattr(sys, "frozen", False) and link.get("args"):
            # the packaged program IS the launcher (both_systems.exe, wherever it sits): start the other system with it
            args = [sys.executable, *link.get("args", []), "--no-browser"]
        else:
            args = [pc._python_for_scripts(settings), str(root / link["script"]), *link.get("args", []), "--no-browser"]
        flags = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW) if sys.platform == "win32" else 0   # hidden console, see scripts/both_systems.py
        env = {k: v for k, v in os.environ.items() if k != "ALGO_SETTINGS_FILE"}
        subprocess.Popen(args, cwd=str(root), env=env, creationflags=flags, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
        for _ in range(60):
            if _port_open(host, port):
                break
            time.sleep(0.5)
        else:
            return _send_json(handler, {"error": f"the {link.get('name', key)} did not start - run {link.get('exe') or link.get('script')} yourself"}, 504)
    handler.send_response(302)
    handler.send_header("Location", link["url"])
    handler.end_headers()


def make_server(settings: Settings, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(settings))
