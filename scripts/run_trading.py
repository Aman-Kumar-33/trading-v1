"""Run the trading system for one session: Angel One live ticks -> candles -> strategies -> ensemble ->
risk manager -> orders. PAPER mode by default (real live data, simulated fills with real costs); LIVE
places real orders through Angel One.

    python scripts/run_trading.py --preflight                 # check everything, trade nothing
    python scripts/run_trading.py                             # paper trading, today's session
    python scripts/run_trading.py --mode live --i-accept-real-money

Safety rules built in:
  - LIVE only trades strategies whose strategy_versions status is LIVE (set them deliberately with
    `scripts/promote_strategies.py --set-live <ids>`); PAPER trades PAPER + LIVE strategies.
  - LIVE refuses to start unless trading.capital in config/settings.yaml is set and Angel One funds cover it.
  - All the risk-manager gates apply: fee hurdles, risk % sizing, caps, daily loss limit -> kill switch,
    max entries per symbol per day, no entries in the first 5 minutes or after 14:45.
  - MACD is long/flat: a bullish cross buys, a bearish cross sells the long - it never shorts. The profit-lock
    and trailing stops are off (trading.profit_lock_stop / trail_stop), so every exit is a MACD cross, a manual
    force-exit from the dashboard, or the safety nets (far 10x-ATR stop, trading.max_carry_days).
  - Intraday longs are bought as CNC: if not exited by the square-off time they are carried overnight.
  - Open positions are saved after every change and restored on the next start (so carried longs are
    managed again next morning). Closed trades go to data/algo.db (trades table).
  - If no tick arrives for trading.heartbeat_seconds during market hours, new entries pause until ticks
    resume (exits keep being managed as soon as prices arrive).
  - Placing orders from your own code requires a static IP registered with Angel One (SEBI rule).

Run it each trading day before 09:15 (e.g. Windows Task Scheduler at 09:00); it exits after the close.
"""
import argparse
import asyncio
import json
import signal as _signal
from datetime import datetime, time, timedelta

from _bootstrap import init

from backend.brokers.angel import AngelOneAdapter
from backend.brokers.angel_feed import AngelTickFeed
from backend.brokers.paper import PaperBroker
from backend.core import events
from backend.core.clock import IST, RealClock
from backend.core.events import EventBus
from backend.core.models import Mode, SessionState, Tick
from backend.core.secrets import get_angel_credentials
from backend.data import angel_instruments as ai
from backend.data.market_calendar import MarketCalendar
from backend.data import chart_history, history_store, trending, watchlist
from backend.data.macd_recorder import MacdRecorder
from backend.data.market_data import MarketDataService
from backend.features.pipeline import build_features
from backend.notify import telegram as tg
from backend.data.session import SessionManager
from backend.execution import process_control as pc
from backend.execution.journal import TradeJournal
from backend.execution.order_manager import OrderManager
from backend.execution.strategy_runner import SimpleRunner
from backend.risk.risk_manager import RiskLimits, RiskManager
from backend.storage.db import init_db
from backend.strategies import ALL_STRATEGIES, INTRADAY_STRATEGIES
from backend.strategies.registry import build_strategy
from backend.core.models import Signal


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["paper", "live"], default="paper")
    p.add_argument("--i-accept-real-money", action="store_true", help="required for --mode live")
    p.add_argument("--preflight", action="store_true", help="validate setup (login, data, strategies) and exit")
    p.add_argument("--symbols", help="comma separated; overrides the stock list saved on the Trading page")
    p.add_argument("--force", action="store_true", help="run even if today isn't a trading day (testing)")
    return p.parse_args()


def _hm(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


async def run(settings, log, args) -> int:
    cfg = settings.section("trading")
    mode = Mode.LIVE if args.mode == "live" else Mode.PAPER
    now = datetime.now(IST)
    calendar = MarketCalendar.from_settings(settings)
    if calendar.session_times(now.date()) is None and not args.force:
        print(f"  {now.date()} is not an NSE trading day - nothing to do")
        return 0

    # ------------------------------------------------------------ strategies - the whole pool, always on (no
    # promotion gate, no allocator/risk weighting - this sandbox system runs every strategy at full power)
    conn = init_db(settings.paths.db_file)
    tf = cfg.get("intraday_timeframe", "15m")
    strategies_by_tf: dict[str, list] = {}
    allowed: set[str] = set()
    for cls in ALL_STRATEGIES:
        strategies_by_tf.setdefault(tf if cls in INTRADAY_STRATEGIES else "1d", []).append(build_strategy(cls, "v1"))
        allowed.add(cls.strategy_id)
    if "1d" in strategies_by_tf:
        print("  a daily-timeframe strategy is in the pool, but this runner only keeps warm-up history for "
              f"{tf} - it would never fire. Remove it from backend/strategies/__init__.py.")
        return 1
    print(f"  mode={mode.value}  strategies: {sorted(allowed)}")

    # ------------------------------------------------------------ capital & risk
    capital = float(cfg.get("capital") or 0) if mode == Mode.LIVE else float(cfg.get("paper_capital", 1_000_000))
    if mode == Mode.LIVE and capital <= 0:
        print("  trading.capital is not set in config/settings.yaml - refusing to trade real money")
        return 1
    limits = RiskLimits(
        capital=capital, intraday_capital_pct=cfg["intraday_capital_pct"], long_term_capital_pct=cfg["long_term_capital_pct"],
        risk_pct_per_trade=cfg["risk_pct_per_trade"], max_concurrent_positions=cfg["max_concurrent_positions"],
        daily_loss_limit_pct=cfg["daily_loss_limit_pct"], max_consecutive_losses=cfg["max_consecutive_losses"],
        max_entries_per_symbol_per_day=cfg["max_entries_per_symbol_per_day"],
        max_cost_to_risk_intraday=cfg["max_cost_to_risk_intraday"], max_cost_to_risk_long_term=cfg["max_cost_to_risk_long_term"],
        min_reward_to_cost=cfg["min_reward_to_cost"], intraday_leverage=cfg["intraday_leverage"])
    risk = RiskManager(limits)
    risk.state.reset_for_new_day(now.date())

    # ------------------------------------------------------------ symbols
    # Only the stocks on the Trading page's stock list (or --symbols / trading.symbols) are MACD-checked. The
    # list is re-read while running, so adding a stock in the dashboard takes effect without a restart.
    state_dir = settings.paths.live_data / "state"
    journal = TradeJournal(conn, state_dir / f"open_positions_{mode.value.lower()}.json", mode)
    watch_stamp = watchlist.change_stamp(settings, args.mode)   # taken BEFORE reading the list
    # The 5 minute system follows the main system: it trades exactly the stocks the main system trades (its trending
    # picks or hand-given list), never picks any itself, and waits here if the main system has not picked yet.
    src = watchlist.follow_settings(settings) or settings   # where today's trending scores come from
    follower = src is not settings
    # No list anywhere = auto mode: the whole index is subscribed (so every stock's bars are built from ticks) and,
    # before the open (once their history is loaded, 09:00-09:20), the top `trending_count` stocks are picked - only those
    # are ever bought, from 09:15.
    auto = watchlist.is_auto_mode(settings, args.mode, args.symbols)
    watch, watch_source = watchlist.active_symbols(settings, args.mode, args.symbols)
    index_symbols = trending.universe_symbols(settings) if auto else []
    held = sorted({d["symbol"] for d in journal.load_positions()})   # open positions still need prices + exits
    if not watch and not held and not auto and not follower:
        print("  no stocks to check - nothing to do.")
        return 0

    # ------------------------------------------------------------ Angel One: login, instrument map
    creds = get_angel_credentials()
    bus = EventBus()
    angel = AngelOneAdapter(bus, creds.api_key, creds.client_code, creds.pin, creds.totp_secret)
    session = angel.login()
    master_cfg = settings.section("angelone")
    master = ai.load_instrument_master(settings.paths.universe / "angel_instrument_master.json",
                                       master_cfg["instrument_master_url"], master_cfg.get("instrument_master_refresh_hours", 24))
    symbol_map = ai.build_symbol_map(master, exchange="NSE")
    missing = [s for s in watch + held if s not in symbol_map]
    watch = [s for s in watch if s in symbol_map]
    index_symbols = [s for s in index_symbols if s in symbol_map]
    symbols = list(dict.fromkeys(watch + [s for s in held if s in symbol_map] + index_symbols))   # everything we need ticks for
    angel.symbol_map = {s: symbol_map[s] for s in symbols}
    if missing:
        print(f"  not in Angel One's instrument master (skipped): {missing}")
    if auto and watchlist.trades_whole_index(settings):
        print(f"  no stock list given - the WHOLE {cfg.get('universe', 'nifty50').upper()} is monitored and traded: {len(watch)} stocks "
              f"(no ranking, no top-N list; trading from {cfg.get('trade_start', '09:15')})")
    elif auto:
        print(f"  no stock list given - AUTO mode: top {int(cfg.get('trending_count', 10))} trending of {len(index_symbols)} "
              f"{cfg.get('universe', 'nifty50')} stocks, picked before the open from their history (trading from 09:15). Today's picks so far: {', '.join(watch) or '(none)'}")
    print(f"  stock list from {watch_source}: {len(watch)} stocks - {', '.join(watch) or '(none)'}")
    if set(held) - set(watch):
        print(f"  also tracking open positions only (exits, no new buys): {', '.join(sorted(set(held) - set(watch)))}")
    if mode == Mode.LIVE:
        funds = angel.funds_sync()
        avail = float(funds.get("availablecash") or funds.get("net") or 0)
        print(f"  Angel One available cash: {avail:,.0f} (configured capital {capital:,.0f})")
        if avail < capital * limits.intraday_capital_pct:
            print("  WARNING: available cash is below the intraday bucket - orders may be rejected")

    if args.preflight:
        print("  preflight OK - login, instrument map and strategies ready (warm-up history is fetched at start). Not trading.")
        return 0

    # ------------------------------------------------------------ engine
    loop = asyncio.get_running_loop()
    broker = angel if mode == Mode.LIVE else PaperBroker(bus, starting_cash=capital)
    if mode == Mode.PAPER:
        await broker.connect()   # PaperBroker tracks last prices from the live tick stream
    square_off = _hm(cfg.get("square_off", "15:10"))
    order_manager = OrderManager(broker, risk, mode=mode, square_off_time=square_off,
                                 carry_longs=cfg.get("carry_longs", False), max_carry_days=cfg.get("max_carry_days", 3),
                                 journal=journal, profit_lock_stop=bool(cfg.get("profit_lock_stop", False)),
                                 trail_stop=bool(cfg.get("trail_stop", False)),
                                 stop_loss_pct=float(cfg.get("stop_loss_pct") or 0) or None)
    order_manager.restore(journal.load_positions())

    def record_signal(sig: Signal, status: str, reject_reason: str | None) -> None:
        conn.execute(
            "INSERT INTO signals (ts, mode, strategy_id, strategy_version, symbol, side, horizon, timeframe, entry,"
            " stop_loss, reason, status, reject_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (sig.ts.isoformat(), mode.value, sig.strategy_id, sig.strategy_version, sig.symbol, sig.side.value,
             sig.horizon.value, sig.timeframe, sig.entry, sig.stop_loss, sig.reason, status, reject_reason))

    min_pct, max_pct = float(cfg.get("min_position_pct", 0.05)), float(cfg.get("max_position_pct", 0.25))
    max_open = int(cfg.get("max_open_positions") or 0)        # 0 = no limit on open positions
    todays = trending.load_today(src, args.mode)
    signal_mode, tolerance = cfg.get("signal_mode", "bar_close"), float(cfg.get("macd_cross_tolerance", 0.0))
    # Research data for the MACD tolerance study: every tick, bar, indicator value and decision for the listed stocks,
    # in <project>/macd_tolerance_data/<date>/ (see backend/data/macd_recorder.py). Turn off with trading.record_research_data.
    recorder = (MacdRecorder(settings.paths.root / cfg.get("research_dir", "macd_tolerance_data"), tf, now.date())
                if cfg.get("record_research_data", True) else None)
    if recorder is not None:
        print(f"  recording research data (ticks, bars, MACD analysis) to {recorder.dir}")
        if todays:
            recorder.save_picks(todays.get("picks", []), datetime.now(IST))
    runner = SimpleRunner(bus, strategies_by_tf, {}, order_manager, capital=capital,
                          history_bars=int(cfg.get("history_bars", 500)), signal_sink=record_signal,
                          deferred_file=state_dir / f"deferred_signals_{mode.value.lower()}.json", watchlist=set(watch),
                          min_position_pct=min_pct, default_position_pct=float(cfg.get("default_position_pct", 0.10)),
                          max_position_pct=max_pct, max_open_positions=max_open, recorder=recorder,
                          square_off_time=None if cfg.get("carry_longs", False) else square_off,
                          signal_mode=signal_mode, tolerance=tolerance,
                          tolerance_atr=float(cfg.get("macd_cross_tolerance_atr", 0.0)),
                          hold_seconds=float(cfg.get("live_hold_seconds", 0.0)),
                          trade_start=_hm(cfg["trade_start"]) if cfg.get("trade_start") else None,
                          stop_loss_pct=float(cfg.get("stop_loss_pct") or 0) or None,
                          trend_scores={p["symbol"]: p["score"] for p in (todays or {}).get("picks", [])} if (auto or follower) else {})
    mds = MarketDataService(bus, calendar, conn, settings.section("candles").get("timeframes", ["1m", "5m", "15m", "1h", "1d"]),
                            settings.paths.live_data / "live", source=f"angel-{mode.value.lower()}")
    session_mgr = SessionManager(calendar, RealClock(), bus, int(settings.section("market").get("eod_freeze_delay_minutes", 5)))

    token_to_symbol = {symbol_map[s]["token"]: s for s in symbols}
    feed = AngelTickFeed(bus, loop, session["jwtToken"], creds.api_key, creds.client_code,
                         angel._client.getfeedToken(), token_to_symbol)

    done = asyncio.Event()

    async def on_session(ev):
        if ev.state == SessionState.POST_CLOSE:
            state_dir.mkdir(parents=True, exist_ok=True)
            trades = order_manager.closed_trades
            summary = {"date": str(ev.trading_date), "mode": mode.value, "trades": len(trades),
                       "net_pnl": round(sum(t.pnl_net for t in trades), 2), "fees": round(sum(t.costs for t in trades), 2),
                       "carried_overnight": [p.symbol for p in order_manager.open_positions],
                       "kill_switch": risk.state.kill_switch.reason or None}
            print(f"  END OF DAY: {json.dumps(summary)}")
            log.info("end of day", extra={"data": summary})
        if ev.state == SessionState.EOD_FROZEN:
            done.set()

    bus.subscribe(events.SESSION, on_session)
    ok, detail = await asyncio.to_thread(
        tg.send_sync, tg.format_startup("TRADING SYSTEM", mode.value, tf, ", ".join(sorted(allowed)), capital, min_pct,
                                        max_pct, max_open, watch, watch_source if not auto or watchlist.trades_whole_index(settings) else "auto - trending",
                                        datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")))
    log.info("startup message to Telegram", extra={"data": {"sent": ok, "detail": detail}})
    feed.start()
    print(f"  live feed started for {len(symbols)} symbols; trading until the close (Ctrl+C to stop)")
    history_days = int(cfg.get("history_days", 60))
    history_lock = asyncio.Lock()   # one fetcher at a time keeps us inside Angel's request rate
    hist_cache: dict = {}           # symbol -> its warm-up bars; the trending picker scores the whole index from these

    async def load_history(syms: list[str], first_call_delay: float = 0.0) -> None:
        """Fetch each symbol's continuous warm-up series from Angel One and hand it to the runner. A symbol
        trades only once its history is loaded - MACD computed over a gappy series gives crosses that don't match
        the chart. Angel's rate limiter rejects some requests; those symbols are retried every 30-120 s until
        they load (see history_store.load_all)."""
        async def fetch_one(sym: str):
            now_ = datetime.now(IST)
            # Before the first bar of the day has closed, a cached series that already holds the whole of the last finished
            # session (pulled after that close - see chart_history.warm_once) is exactly what Angel would return now: use it
            # and start trading at once instead of spending ~4 s per stock. A start later in the session needs today's
            # bars as well, so that still fetches.
            cached = history_store.load_history(settings, sym, tf)
            if chart_history.usable_at_start(settings, cached, tf, now_):
                cached.attrs["from_cache"] = True
                return cached
            return await asyncio.to_thread(history_store.refresh_symbol, settings, angel, sym, symbol_map[sym],
                                           tf, history_days, now_)
        def loaded(sym: str, df) -> None:
            hist_cache[sym] = df
            runner.set_history(sym, tf, df)
            if loop.is_running():   # a listed stock whose history arrived after its bar closed: take a fresh cross, record its bars
                asyncio.ensure_future(runner.catch_up_entries([sym], datetime.now(IST))).add_done_callback(
                    lambda _f: runner.backfill_recording([sym]))

        await history_store.load_all(syms, fetch_one, loaded, done, history_lock, first_delay=first_call_delay)

    history_tasks = [asyncio.create_task(load_history(list(symbols), first_call_delay=3))]

    async def track(new_symbols: list[str]) -> None:
        """Start ticks + warm-up history for stocks the process wasn't watching when it started."""
        fresh = [s for s in new_symbols if s not in token_to_symbol.values()]
        usable = [s for s in fresh if s in symbol_map]
        if len(usable) < len(fresh):
            print(f"  not in Angel One's instrument master (ignored): {[s for s in fresh if s not in symbol_map]}")
        if not usable:
            return
        for s in usable:
            angel.symbol_map[s] = symbol_map[s]
        feed.subscribe_more({symbol_map[s]["token"]: s for s in usable})
        history_tasks.append(asyncio.create_task(load_history(usable)))

    async def refresh_active(why: str) -> None:
        """Point the runner at the stocks allowed to be bought now (hand-given list, or today's trending picks) and
        make sure they - and, in auto mode, the whole index - have ticks and history."""
        new, source = watchlist.active_symbols(settings, args.mode, args.symbols)
        todays_picks = trending.load_today(src, args.mode)
        runner.set_watchlist(new)
        runner.set_trend_scores({p["symbol"]: p["score"] for p in (todays_picks or {}).get("picks", [])}
                                if (watchlist.is_auto_mode(settings, args.mode, args.symbols) or follower) else {})
        added = [s for s in new if s not in token_to_symbol.values()]
        log.info("stock list changed", extra={"data": {"why": why, "source": source, "stocks": new, "newly_subscribed": added}})
        print(f"  stock list updated ({why}): {len(new)} stocks - {', '.join(new) or '(none)'}")
        await runner.catch_up_entries(new, datetime.now(IST))   # a fresh cross on the bar that closed just before the pick
        runner.backfill_recording(new)   # bars of today that closed before these stocks were on the list
        await track(added)
        if watchlist.is_auto_mode(settings, args.mode, args.symbols):
            await track([s for s in trending.universe_symbols(settings) if s in symbol_map])

    async def apply_watchlist_change() -> None:
        nonlocal watch_stamp
        stamp = watchlist.change_stamp(settings, args.mode)
        if stamp == watch_stamp:
            return
        watch_stamp = stamp
        await refresh_active("stock box changed")

    pick_retry_at = datetime.min.replace(tzinfo=IST)

    async def pick_trending(now: datetime) -> None:
        """Rank the index and fix today's trending list. Done before the open so trading starts at 09:15: as soon as the
        index's warm-up history is loaded (history takes ~4 s per stock), or at 09:20 with whatever has loaded. Before
        09:30 only the history is ranked - there is no day yet to look at; a late restart also uses today's quote."""
        nonlocal pick_retry_at
        if now < pick_retry_at:
            return
        index = [s for s in trending.universe_symbols(settings) if s in symbol_map]
        ready = {s: hist_cache[s] for s in index if s in hist_cache}
        go, use_quotes = trending.pick_plan(now, started_at, len(ready), len(index))
        if not go:
            return
        quotes: dict = {}
        if use_quotes:
            try:
                token_of = {symbol_map[s]["token"]: s for s in index}
                rows = await asyncio.to_thread(angel.quotes_full_sync, {"NSE": list(token_of)})
                quotes = {token_of[t]: row for t, row in rows.items() if t in token_of}
            except Exception as exc:  # noqa: BLE001 - rank on history alone rather than not trading at all
                log.error("market quote for the trending picker failed", extra={"data": {"error": repr(exc)}})
                if now - started_at < timedelta(minutes=3) and now.time() < time(9, 40):
                    pick_retry_at = now + timedelta(seconds=20)
                    return
        picks = trending.rank_picks(ready, quotes, int(cfg.get("trending_count", 10)), max_price=max_pct * capital)
        if not picks:
            log.error("trending picker found nothing to rank", extra={"data": {"history_loaded": len(ready)}})
            pick_retry_at = now + timedelta(seconds=60)
            return
        trending.save_picks(settings, args.mode, picks, now.date(), tf)
        if recorder is not None:
            recorder.save_picks(picks, now)
        print(f"  TRENDING TOP {len(picks)} (of {len(ready)} ranked): " + ", ".join(f"{p['symbol']}({p['score']:.0f})" for p in picks))
        log.info("trending stocks picked", extra={"data": {"picks": picks, "quotes_used": bool(quotes)}})
        await refresh_active("trending picked")
        tg.notify(tg.format_picks(mode.value, tf, picks, now.strftime("%Y-%m-%d %H:%M:%S")))

    async def save_official_bars() -> None:
        """After the close: fetch Angel's official 15m bars for every recorded stock, save them with the MACD analysis
        and write the live-vs-official comparison (tick-built bars can differ slightly from the exchange's)."""
        if recorder is None or datetime.now(IST).time() < time(15, 30):
            return   # stopped early (Ctrl+C / mid-day) - the day's official bars are not complete
        for t in history_tasks:
            t.cancel()
        rows = []
        for sym in sorted(recorder.symbols & set(symbol_map)):
            try:
                df = await asyncio.to_thread(history_store.refresh_symbol, settings, angel, sym, symbol_map[sym], tf,
                                             history_days, datetime.now(IST))
                rows.append(recorder.save_official(sym, build_features(df, tf, core_only=True)))
            except Exception as exc:  # noqa: BLE001 - the live data is already saved; this is an extra
                log.error("official bars not saved", extra={"data": {"symbol": sym, "error": repr(exc)}})
            await asyncio.sleep(history_store.CALL_SPACING_SECONDS)
        recorder.save_comparison(rows)
        print(f"  research data saved: {recorder.dir}")

    heartbeat = timedelta(seconds=int(cfg.get("heartbeat_seconds", 120)))
    started_at = datetime.now(IST)
    try:
        loop.add_signal_handler(_signal.SIGINT, done.set)
    except NotImplementedError:
        pass   # Windows: Ctrl+C raises KeyboardInterrupt instead, handled below
    reconnect_cooldown = timedelta(seconds=30)
    last_reconnect_attempt = None
    try:
        while not done.is_set():
            try:
                now = datetime.now(IST)
                await session_mgr.step(now)
                await mds.on_clock(now)
                await order_manager.on_time(now)   # time-driven square-off - runs even if the feed is dead
                await apply_watchlist_change()
                if watchlist.ranks_stocks(settings, args.mode, args.symbols) and trending.load_today(settings, args.mode) is None:
                    await pick_trending(now)
                for cmd in pc.pop_commands(settings, args.mode):
                    if cmd.get("action") == "close_position" and cmd.get("symbol"):
                        await order_manager.close_manual(cmd["symbol"], now)
                        if order_manager.has_position(cmd["symbol"]):
                            # the exit order failed (e.g. no tick yet for this symbol) - keep the request so
                            # the next loop retries instead of silently dropping it
                            pc.queue_close_position(settings, args.mode, cmd["symbol"])
                in_hours = time(9, 15) <= now.time() < time(15, 25) and now - started_at > heartbeat
                stale = in_hours and (feed.last_tick_at is None or now - feed.last_tick_at > heartbeat)
                if stale and risk.data_feed_ok:
                    log.critical("no ticks received - new entries paused", extra={"data": {"last_tick_at": str(feed.last_tick_at)}})
                elif not stale and not risk.data_feed_ok:
                    log.warning("market data feed back - entries resumed")
                risk.data_feed_ok = not stale
                if stale and in_hours and (last_reconnect_attempt is None or now - last_reconnect_attempt > reconnect_cooldown):
                    last_reconnect_attempt = now
                    log.warning("attempting feed reconnect", extra={"data": {"last_tick_at": str(feed.last_tick_at)}})
                    try:
                        feed.stop()
                        feed = AngelTickFeed(bus, loop, session["jwtToken"], creds.api_key, creds.client_code,
                                             angel._client.getfeedToken(), token_to_symbol)
                        feed.start()
                        exch_tokens: dict[str, list[str]] = {}
                        for s in token_to_symbol.values():
                            exch_tokens.setdefault(symbol_map[s]["exchange"], []).append(symbol_map[s]["token"])
                        ltp_by_token = await asyncio.to_thread(angel.ltp_bulk_sync, exch_tokens)
                        refreshed_at = datetime.now(IST)
                        for token, ltp in ltp_by_token.items():
                            sym = token_to_symbol.get(token)
                            if sym:
                                await bus.publish(events.TICK, Tick(symbol=sym, ts=refreshed_at, ltp=ltp))
                        log.warning("reconnected - refreshed prices before resuming",
                                   extra={"data": {"symbols_refreshed": len(ltp_by_token)}})
                    except Exception as exc:  # noqa: BLE001 - a failed reconnect attempt must not kill the session
                        log.error("feed reconnect failed - will retry", extra={"data": {"error": repr(exc)}})
            except Exception as exc:  # noqa: BLE001 - one bad tick/handler must not take the whole session down
                log.exception("main loop iteration failed - continuing", extra={"data": {"error": repr(exc)}})
            try:
                await asyncio.wait_for(done.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass
        await save_official_bars()
    finally:
        for t in history_tasks:
            t.cancel()
        feed.stop()
        if recorder is not None:
            recorder.close()
        conn.close()
    return 0


def main() -> int:
    args = parse_args()
    if args.mode == "live" and not args.i_accept_real_money:
        print("  --mode live places REAL orders with real money. Re-run with --i-accept-real-money.")
        return 1
    settings, log = init(f"run_trading_{args.mode}")
    try:
        return asyncio.run(run(settings, log, args))
    except KeyboardInterrupt:
        print("  stopped by user - open positions were saved and will be restored next start")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
