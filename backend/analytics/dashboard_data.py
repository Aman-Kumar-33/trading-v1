"""Gathers data from disk (backtest/full-system results, the `trades` DB table, state files, logs) into
the plain-dict shapes the dashboard API returns as JSON. Kept separate from the HTTP server so that
module stays pure routing, and separate from backend.analytics.metrics so the maths stays reusable by
anything else (a future CLI report, the weekly evolution loop) without pulling in file/DB access.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from backend.analytics.metrics import benchmark_daily_returns, compute_metrics
from backend.core.clock import IST
from backend.core.config import Settings
from backend.data import chart_history
from backend.strategies import ALL_STRATEGIES
from backend.strategies.registry import get_status

# strategy_id -> "INTRADAY" | "LONG_TERM", so a log line can be bucketed even though it only ever
# mentions the strategy_id (in data.strategy), never the horizon itself.
STRATEGY_HORIZON = {cls.strategy_id: cls.horizon.value for cls in ALL_STRATEGIES}


def _trades_from_df(df: pd.DataFrame | None) -> list:
    if df is None or df.empty or not {"pnl_gross", "costs", "pnl_net"} <= set(df.columns):
        return []
    exit_col = "exit_ts" if "exit_ts" in df.columns else "closed_at" if "closed_at" in df.columns else None
    out = []
    for row in df.itertuples(index=False):
        d = row._asdict()
        ns = SimpleNamespace(pnl_gross=d["pnl_gross"], costs=d["costs"], pnl_net=d["pnl_net"])
        ns.exit_ts = d.get(exit_col) if exit_col else None
        out.append(ns)
    return out


def _trades_from_db_rows(rows) -> list:
    return [SimpleNamespace(pnl_gross=r["pnl_gross"] or 0.0, costs=r["costs"] or 0.0, pnl_net=r["pnl_net"] or 0.0,
                           exit_ts=r["exit_ts"]) for r in rows if r["exit_ts"] is not None]


def benchmark_returns(settings: Settings) -> pd.Series | None:
    path = settings.paths.clean / "1d" / "IDX_NSEI.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return benchmark_daily_returns(df) if not df.empty else None


def load_capital(settings: Settings) -> dict:
    cfg = settings.section("trading")
    return {"capital": float(cfg.get("capital") or 0), "paper_capital": float(cfg.get("paper_capital", 1_000_000)),
           "intraday_pct": cfg.get("intraday_capital_pct"), "long_term_pct": cfg.get("long_term_capital_pct")}


def market_status(settings: Settings) -> dict:
    """Pure function of calendar + wall clock - no live bus/process needed, so the dashboard can show
    this even when nothing is running. "is_open" means actively in a trading session (OPEN/SQUARE_OFF);
    everything else (closed, pre-open, post-close, EOD) is when the live/paper tab should show analysis
    instead of controls."""
    from backend.core.clock import RealClock
    from backend.data.market_calendar import MarketCalendar
    from backend.data.session import SessionManager

    now = RealClock().now()
    sm = SessionManager(MarketCalendar.from_settings(settings), RealClock(), bus=None)
    state = sm.state_at(now)
    return {"state": state.value, "is_open": state.value in ("OPEN", "SQUARE_OFF"), "ts": now.isoformat()}


def db_trade_rows(conn: sqlite3.Connection, mode: str, limit: int = 500, symbol: str | None = None) -> list[dict]:
    base = """select symbol, strategy_id, horizon, side, qty, entry_ts, entry_price, exit_ts,
                    exit_price, stop_loss, target, pnl_gross, costs, pnl_net, r_multiple, exit_reason
             from trades where mode=? and exit_ts is not null"""
    args: list = [mode.upper()]
    if symbol:
        base += " and symbol=?"
        args.append(symbol)
    rows = conn.execute(base + " order by exit_ts desc limit ?", (*args, limit)).fetchall()
    return [dict(r) for r in rows]


def trading_timeframe(settings: Settings) -> str:
    return settings.section("trading").get("intraday_timeframe", "15m")


def trading_config(settings: Settings) -> dict:
    """The knobs the Trading page shows next to the controls."""
    cfg = settings.section("trading")
    return {"timeframe": trading_timeframe(settings), "capital": float(cfg.get("paper_capital", 1_000_000)),
            "min_position_pct": cfg.get("min_position_pct", 0.05), "max_position_pct": cfg.get("max_position_pct", 0.25),
            "max_open_positions": int(cfg.get("max_open_positions") or 0), "trending_count": cfg.get("trending_count", 10),
            "whole_index": str(cfg.get("trade_universe", "")).strip().lower() in ("all", "index", "whole"),
            "universe": cfg.get("universe", "nifty50"),
            "signal_mode": cfg.get("signal_mode", "bar_close"), "tolerance": float(cfg.get("macd_cross_tolerance", 0.0)),
            "tolerance_atr": float(cfg.get("macd_cross_tolerance_atr", 0.0)), "hold_seconds": float(cfg.get("live_hold_seconds", 0.0)),
            "trade_start": cfg.get("trade_start"), "square_off": cfg.get("square_off"), "stop_loss_pct": cfg.get("stop_loss_pct"),
            "instance": {"name": settings.section("instance").get("name", ""),
                         "follows": bool(settings.section("instance").get("follow")),
                         "links": settings.section("instance").get("links", [])}}


def trending_view(settings: Settings, mode: str) -> dict:
    """Everything the Trading page's trending panel needs: is auto mode on, today's picks, what is in use."""
    from backend.data import trending, watchlist
    src = watchlist.follow_settings(settings)      # the 5 minute system shows the main system's stocks
    if src is not None:
        view = trending_view(src, mode)
        view.update({"follows": True, "using": "the main system's stocks - " + view["using"],
                     "stocks_now": watchlist.active_symbols(settings, mode)[0]})
        return view
    given, source = watchlist.resolve_symbols(settings, mode)
    today = trending.load_today(settings, mode)
    whole = watchlist.trades_whole_index(settings) and not given
    if whole:
        stocks = watchlist.active_symbols(settings, mode)[0]
        return {"auto": True, "whole_index": True, "using": f"the whole {str(settings.section('trading').get('universe', 'nifty50')).upper()}",
                "stock_list": [], "stocks_now": stocks, "date": None, "picked_at": None, "timeframe": trading_timeframe(settings), "picks": [],
                "count": len(stocks), "universe": settings.section("trading").get("universe", "nifty50")}
    return {"auto": not given, "whole_index": False, "using": "your stock list" if given else "today's trending stocks",
            "stock_list": given, "date": (today or {}).get("date"), "picked_at": (today or {}).get("picked_at"),
            "timeframe": (today or {}).get("timeframe", trading_timeframe(settings)), "picks": (today or {}).get("picks", []),
            "count": int(settings.section("trading").get("trending_count", 10)),
            "universe": settings.section("trading").get("universe", "nifty50")}


def symbol_candles(settings: Settings, conn: sqlite3.Connection, symbol: str, timeframe: str = "5m",
                   limit: int = 300) -> pd.DataFrame:
    """Historical candles for context, plus any more-recent closed candles already written to the
    live `candles` table (populated during paper/live trading - see backend/data/market_data.py). There
    is no forming/in-progress bar available cross-process (see that module's docstring) - this is "last
    closed bar, polled", not true tick-by-tick streaming.

    The history of every timeframe is Angel One's own series (backend/data/chart_history.py: cached, built from a
    finer cached series, or fetched - never the old data lake), and its bars win over live-built duplicates - so
    for the trading timeframe the chart, the MACD drawn under it and the signals the system acted on all come from
    the same numbers."""
    def live_bars(tf: str, after: str) -> pd.DataFrame:
        return chart_history.live_rows(settings, symbol, tf, after, own_conn=conn)

    hist, stale = chart_history.chart_history_ex(settings, symbol, timeframe, limit, live=live_bars, blocking=False)
    hist = hist[["ts", "open", "high", "low", "close", "volume"]] if hist is not None else pd.DataFrame(
        columns=["ts", "open", "high", "low", "close", "volume"])
    # no history at all (Angel refused and nothing is cached): show only the bars of the latest session(s), never
    # a few stray live bars from days the system happened to run, which would look like a hole in the chart
    last_ts = str(hist["ts"].iloc[-1]) if not hist.empty else str(chart_history.last_finished_session(settings, datetime.now(IST)))
    live = live_bars(timeframe, last_ts)
    out = pd.concat([hist, live], ignore_index=True) if not live.empty else hist
    if out.empty:
        out.attrs["stale"] = stale
        return out
    parsed = pd.to_datetime(out["ts"], format="mixed", utc=True).dt.tz_convert("Asia/Kolkata")
    if timeframe == "1d":
        # a daily bar still being built live can get stored with the session-open time (09:15) instead
        # of midnight - normalize so it doesn't land out of order among the historical midnight-ts rows
        # (lightweight-charts requires strictly ascending time, a single out-of-order bar breaks the chart)
        parsed = parsed.dt.normalize()
    out = out.assign(_t=parsed).sort_values("_t", kind="stable").drop_duplicates("_t", keep="first") \
        .tail(limit).drop(columns="_t").reset_index(drop=True)
    out["ts"] = out["ts"].astype(str)
    for col in ("open", "high", "low", "close"):
        out[col] = out[col].astype(float)
    out["volume"] = out["volume"].fillna(0).astype(int)
    if timeframe == trading_timeframe(settings):
        out = with_own_closes(out, chart_history.live_rows(settings, symbol, timeframe, "", own_conn=conn, own_only=True))
    out.attrs["stale"] = stale          # behind the clock: the missing bars are being fetched, ask again in a moment
    return out


def with_own_closes(out: pd.DataFrame, own: pd.DataFrame) -> pd.DataFrame:
    """For the system's own timeframe, the bars the trading process built itself from its ticks decide its trades - and their
    closes differ from Angel's official ones by a few paise to a few rupees (the last tick before the bar ended vs the
    exchange's last trade). On a borderline MACD cross that moves the arrow by one candle. So the MACD the chart draws, and the
    arrows on it, use the system's own closes wherever it recorded a bar; bars from before it started stay Angel's. The
    closing bars (15:15 on) are left alone - they are stored in the settled form (see history_store.settle_closing_bars)."""
    if own.empty or out.empty:
        return out
    def key(col):
        return pd.to_datetime(col, format="mixed", utc=True)
    own = own.assign(_k=key(own["ts"]))
    local = own["_k"].dt.tz_convert("Asia/Kolkata")
    own = own[(local.dt.hour * 60 + local.dt.minute) < 15 * 60 + 15].drop_duplicates("_k", keep="last")
    new = key(out["ts"]).map(dict(zip(own["_k"], own["close"].astype(float))))
    hit = new.notna()
    if not hit.any():
        return out
    out = out.copy()
    out.loc[hit, "close"] = new[hit].astype(float)
    out.loc[hit, "high"] = out.loc[hit, ["high", "close"]].max(axis=1)
    out.loc[hit, "low"] = out.loc[hit, ["low", "close"]].min(axis=1)
    return out


def chart_context(settings: Settings, mode: str, symbol: str) -> dict:
    """What a person needs to read the arrows on a chart: the arrows show what the MACD rule says for ANY stock, but an order
    is only placed while this system is running and only for the stocks on its list (and inside its trading hours)."""
    from backend.data import watchlist
    from backend.execution import process_control as pc
    cfg = settings.section("trading")
    try:
        listed = symbol in watchlist.active_symbols(settings, mode)[0]
    except Exception:  # noqa: BLE001 - a missing list file must not break a chart
        listed = False
    return {"running": bool(pc.status(settings, mode).get("running")), "listed": listed,
            "system": settings.section("instance").get("name") or "main", "signal_mode": cfg.get("signal_mode", "bar_close"),
            "trade_start": cfg.get("trade_start"), "square_off": cfg.get("square_off")}


def macd_overlay(candles: pd.DataFrame, show_last: int, confirm_tolerance: tuple[float, float] | None = None,
                 square_off: str | None = None, tf_minutes: int | None = None) -> dict:
    """MACD (12, 26, 9) over the WHOLE series passed in, then trimmed to the last `show_last` bars - the
    warm-up bars before the visible window are what make the first visible values right. `signals` are the
    cross bars exactly as backend.strategies.intraday.macd_only detects them (bar `i-1` at/below, bar `i`
    above), so what the chart marks is what the strategy fires on. With `confirm_tolerance` = (floor, x ATR) - the 5 minute
    system's "candle_confirmed" mode - a BUY is marked only on the closed bar where the cross has cleared the tolerance
    (the bar the system buys on); SELL stays the plain cross down. With `square_off` ("15:00") and the timeframe's minutes, a SELL
    whose order time (the candle's close) is at/after the square-off is not marked: everything is already sold then, the exit on the
    chart is the square-off itself (the OUT marker), and a later cross down changes nothing."""
    from backend.features import indicators as ind
    from backend.strategies.intraday.macd_only import confirmed_cross_up, macd_cross_masks

    if candles.empty:
        return {"candles": candles, "macd": [], "signals": []}
    m = ind.macd(candles["close"].astype(float))
    macd, sig = m["macd"].to_numpy(dtype=float), m["signal"].to_numpy(dtype=float)
    bull, bear = macd_cross_masks(macd, sig)
    if confirm_tolerance is not None:
        floor, mult = confirm_tolerance
        atr = ind.atr(candles["high"].astype(float), candles["low"].astype(float), candles["close"].astype(float), 14).to_numpy(dtype=float)
        gap = macd - sig
        bull = np.zeros(len(candles), dtype=bool)
        for i in range(max(0, len(candles) - show_last - 1), len(candles)):
            tol = max(floor, mult * atr[i]) if atr[i] == atr[i] and atr[i] > 0 else floor
            bull[i] = confirmed_cross_up(gap[:i + 1], tol)
    start = max(0, len(candles) - show_last)
    cut = None
    if square_off and tf_minutes and tf_minutes < 240:
        cut = int(square_off[:2]) * 60 + int(square_off[3:5])
    rows, signals = [], []
    for i in range(start, len(candles)):
        if macd[i] != macd[i] or sig[i] != sig[i]:
            continue
        ts = candles["ts"].iloc[i]
        rows.append({"ts": ts, "macd": round(float(macd[i]), 4), "signal": round(float(sig[i]), 4),
                     "hist": round(float(macd[i] - sig[i]), 4)})
        if bear[i] and not bull[i] and cut is not None:
            at = pd.Timestamp(ts)
            if at.hour * 60 + at.minute + tf_minutes >= cut:
                continue
        if bull[i] or bear[i]:
            signals.append({"ts": ts, "side": "BUY" if bull[i] else "SELL", "close": float(candles["close"].iloc[i]),
                            "macd": round(float(macd[i]), 4), "signal": round(float(sig[i]), 4)})
    return {"candles": candles.iloc[start:].reset_index(drop=True), "macd": rows, "signals": signals}


def db_trades(conn: sqlite3.Connection, mode: str, strategy_id: str | None = None,
              since: str | None = None) -> list:
    where, args = "mode=?", [mode.upper()]
    if strategy_id:
        where += " and strategy_id=?"
        args.append(strategy_id)
    if since:
        where += " and exit_ts>=?"
        args.append(since)
    rows = conn.execute(f"select pnl_gross, costs, pnl_net, exit_ts from trades where {where}", args).fetchall()
    return _trades_from_db_rows(rows)


def today_ist() -> str:
    from backend.core.clock import RealClock
    return RealClock().now().date().isoformat()


def daily_pnl(conn: sqlite3.Connection, mode: str, days: int = 30) -> list[dict]:
    """Net P&L grouped by calendar day (IST exit date) - what a day-by-day hedge-fund-style P&L chart
    needs, which `overview()`'s single running total can't show."""
    rows = conn.execute("""select substr(exit_ts, 1, 10) as day, sum(pnl_gross) as gross, sum(costs) as costs,
                                 sum(pnl_net) as net, count(*) as trades
                          from trades where mode=? and exit_ts is not null
                          group by day order by day desc limit ?""", (mode.upper(), days)).fetchall()
    return [dict(r) for r in reversed(rows)]


def latest_backtest_trades(settings: Settings, strategy_id: str) -> list:
    """Prefer the newest full-system run (live-parity fills/costs) over the older per-strategy backtest -
    both write trades.parquet with the same SimTrade columns, so the same loader handles either."""
    root = settings.paths.results / "full_system"
    best_path, best_mtime = None, -1.0
    if root.exists():
        for run_dir in root.iterdir():
            p = run_dir / "trades.parquet"
            if p.exists() and p.stat().st_mtime > best_mtime:
                best_path, best_mtime = p, p.stat().st_mtime
    if best_path is not None:
        df = pd.read_parquet(best_path)
        return _trades_from_df(df[df["strategy_id"] == strategy_id] if "strategy_id" in df.columns else df)
    p = settings.paths.results / strategy_id / "v1" / "trades.parquet"
    return _trades_from_df(pd.read_parquet(p)) if p.exists() else []


def strategy_rows(settings: Settings, conn: sqlite3.Connection) -> list[dict]:
    bench = benchmark_returns(settings)
    capital = load_capital(settings)
    live_cap = capital["capital"] or capital["paper_capital"]
    out = []
    for cls in ALL_STRATEGIES:
        sid = cls.strategy_id
        bt = compute_metrics(latest_backtest_trades(settings, sid), capital["paper_capital"], bench)
        paper = compute_metrics(db_trades(conn, "paper", sid), capital["paper_capital"], bench)
        live = compute_metrics(db_trades(conn, "live", sid), live_cap, bench)
        out.append({"id": sid, "horizon": cls.horizon.value, "status": get_status(conn, sid, "v1") or "RESEARCH",
                   "backtest": bt.as_dict(), "paper": paper.as_dict(), "live": live.as_dict()})
    return out


def overview(settings: Settings, conn: sqlite3.Connection) -> dict:
    bench = benchmark_returns(settings)
    capital = load_capital(settings)
    today = today_ist()
    paper = compute_metrics(db_trades(conn, "paper"), capital["paper_capital"], bench)
    live = compute_metrics(db_trades(conn, "live"), capital["capital"] or capital["paper_capital"], bench)
    paper_today = compute_metrics(db_trades(conn, "paper", since=today), capital["paper_capital"], bench)
    live_today = compute_metrics(db_trades(conn, "live", since=today), capital["capital"] or capital["paper_capital"], bench)
    return {"capital": capital, "paper": paper.as_dict(), "live": live.as_dict(),
           "paper_today": paper_today.as_dict(), "live_today": live_today.as_dict()}


def ensemble_weights(settings: Settings, mode: str) -> dict:
    path = settings.paths.live_data / "state" / f"tracker_{mode}.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for key, trades in payload.get("history", {}).items():
        sid, regime = key.split("|", 1)
        n = len(trades)
        mean_r = sum(trades) / n if n else 0.0
        out.setdefault(sid, {})[regime] = {"n": n, "mean_r": round(mean_r, 4)}
    return out


def process_status(settings: Settings) -> dict:
    out = {}
    for mode in ("paper", "live"):
        path = settings.paths.live_data / "state" / f"trading_process_{mode}.json"
        if path.exists():
            try:
                out[mode] = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                out[mode] = {"running": False}
        else:
            out[mode] = {"running": False}
    return out


def open_positions(settings: Settings, mode: str) -> list:
    path = settings.paths.live_data / "state" / f"open_positions_{mode}.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def positions_with_pnl(settings: Settings, conn: sqlite3.Connection, mode: str) -> list:
    """Open positions enriched with the latest closed-candle price and unrealized P&L - without this, the
    dashboard only ever shows the entry price, so a position's current standing (like the TRENT/AXISBANK
    cases) can only be worked out by hand against a separate price source."""
    positions = open_positions(settings, mode)
    for p in positions:
        row = conn.execute("select close from candles where symbol=? order by ts_start desc limit 1",
                          (p["symbol"],)).fetchone()
        ltp = float(row["close"]) if row else None
        p["ltp"] = ltp
        if ltp is not None:
            direction = 1 if p["side"] == "BUY" else -1
            p["unrealized_pnl"] = round(direction * (ltp - p["entry_price"]) * p["qty"], 2)
        else:
            p["unrealized_pnl"] = None
    return positions


def _parse_log_line(line: str) -> dict:
    """One structured entry: ts/level/logger/msg kept as-is, `data` flattened out, plus `horizon`
    classified from data.strategy where possible (None for lines with no strategy - session/feed/system
    events - the frontend buckets those separately rather than dropping them)."""
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        return {"raw": line}
    data = row.get("data") or {}
    strategy_id = data.get("strategy") or data.get("strategy_id")
    return {"ts": row.get("ts"), "level": row.get("level"), "logger": row.get("logger"), "msg": row.get("msg"),
           "data": data, "horizon": STRATEGY_HORIZON.get(strategy_id)}


def tail_log(settings: Settings, name: str, n: int = 200) -> list[dict]:
    path = settings.paths.logs / f"{name}.jsonl"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return [_parse_log_line(ln.rstrip("\n")) for ln in lines[-n:] if ln.strip()]


def pipeline_events(settings: Settings, name: str, since: str | None = None, lines: int = 500) -> list[dict]:
    """Feeds the live ensemble animation / engine feed - same parsed shape as tail_log, just newer-than-
    `since` (an ISO timestamp string; plain lexicographic compare works since ISO-8601 sorts correctly)."""
    entries = tail_log(settings, name, lines)
    if since:
        entries = [e for e in entries if e.get("ts") and e["ts"] > since]
    return entries
