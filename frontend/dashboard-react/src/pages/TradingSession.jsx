import { useEffect, useRef, useState } from "react";
import { api, fmt, cls, forceExit } from "../api.js";
import { toast } from "../toast.js";
import EnsembleFlow from "../components/EnsembleFlow.jsx";
import { LogEntry } from "../components/logs.jsx";
import { EquityCurve } from "../components/charts.jsx";
import { PositionsTable, TradesTable } from "../components/blotter.jsx";
import WatchlistBox from "../components/WatchlistBox.jsx";
import TrendingBox from "../components/TrendingBox.jsx";

const LIVE_PHRASE = "I_ACCEPT_REAL_MONEY_RISK";

function StartPanel({ mode, onStarted, stockCount, manualStart }) {
  const [confirmText, setConfirmText] = useState("");
  const isLive = mode === "live";

  async function start() {
    if (isLive && !window.confirm("This places REAL orders with REAL money. Continue?")) return;
    try {
      const body = {};
      if (isLive) body.confirm = LIVE_PHRASE;
      await api(`/api/${mode}/start`, body);
      toast(`${mode} trading started`);
      onStarted();
    } catch (e) {
      toast(e.message, true);
    }
  }

  return (
    <div>
      {isLive && (
        <div className="warning-banner">
          This places REAL orders with REAL money through Angel One. Only strategies set to LIVE status
          will trade. Review Paper Trading results first.
        </div>
      )}
      {isLive && (
        <div className="controls">
          <label>Type <code>{LIVE_PHRASE}</code> to enable Start:</label>
          <input value={confirmText} onChange={(e) => setConfirmText(e.target.value)} placeholder="confirmation phrase" />
        </div>
      )}
      <div className="controls">
        <button className={isLive ? "danger" : ""} disabled={isLive && confirmText !== LIVE_PHRASE} onClick={start}>
          Start {isLive ? "LIVE" : "Paper"} Trading
        </button>
        {manualStart && <span className="hint">this system never starts by itself - press the button to start paper trading</span>}
        {!manualStart && stockCount === 0 && <span className="hint">stock box is empty - the system {status?.config?.whole_index ? "trades the whole index" : "will pick today's trending stocks itself"}</span>}
      </div>
    </div>
  );
}

function LiveSession({ mode, strategies, onStopped, onSelectSymbol }) {
  const [feed, setFeed] = useState([]);
  const [newEvents, setNewEvents] = useState([]);
  const [positions, setPositions] = useState([]);
  const [tape, setTape] = useState([]);
  const sinceRef = useRef(null);

  useEffect(() => {
    let alive = true;
    async function poll() {
      try {
        const qs = sinceRef.current ? `&since=${encodeURIComponent(sinceRef.current)}` : "";
        const [events, pos, trades] = await Promise.all([
          api(`/api/events?mode=${mode}${qs}`),
          api(`/api/positions?mode=${mode}`),
          api(`/api/trades?mode=${mode}&limit=12`),
        ]);
        if (!alive) return;
        if (events.length) {
          sinceRef.current = events[events.length - 1].ts;
          setFeed((f) => [...f, ...events].slice(-200));
        }
        setNewEvents(events);
        setPositions(pos);
        setTape(trades);
      } catch { /* transient - next poll retries */ }
    }
    poll();
    const id = setInterval(poll, 1500);
    return () => { alive = false; clearInterval(id); };
  }, [mode]);

  async function stop() {
    await api(`/api/${mode}/stop`, {});
    toast(`${mode} trading stopped`);
    onStopped();
  }

  return (
    <div>
      <div className="controls">
        <span className="status-badge running">running</span>
        <button className="danger" onClick={stop}>Stop {mode}</button>
      </div>
      <p className="section-label">Ensemble - live</p>
      <EnsembleFlow strategies={strategies} events={newEvents} />
      <p className="section-label">Live Engine Feed</p>
      <div className="engine-feed">
        {feed.length === 0 && <div className="log-empty">waiting for activity...</div>}
        {feed.slice().reverse().map((e, i) => (
          <div key={i} className={`feed-line level-${e.level}`}><LogEntry e={e} /></div>
        ))}
      </div>
      <p className="section-label">Live Trade Tape</p>
      <div className="trade-tape">
        {tape.length === 0 && <span className="hint">no fills yet</span>}
        {tape.map((t, i) => (
          <span key={i} className={`tape-item ${cls(t.pnl_net)}`} onClick={() => onSelectSymbol?.(t.symbol, mode)}
               style={{ cursor: onSelectSymbol ? "pointer" : "default" }}>{t.symbol} {fmt(t.pnl_net)}</span>
        ))}
      </div>
      <p className="section-label">Open Positions</p>
      <PositionsTable positions={positions} onSelect={(s) => onSelectSymbol?.(s, mode)}
                      onForceExit={(s) => forceExit(s, mode)} />
    </div>
  );
}

// Open positions + force exit while NO trading process is running (before the open, after a stop) - these
// used to be invisible here, so a carried position couldn't be exited without starting trading first.
function OpenPositions({ mode, onSelectSymbol }) {
  const [positions, setPositions] = useState([]);
  const load = () => api(`/api/positions?mode=${mode}`).then(setPositions).catch(() => {});
  useEffect(() => { load(); const id = setInterval(load, 5000); return () => clearInterval(id); }, [mode]);
  return (
    <>
      <p className="section-label">Open positions ({positions.length})</p>
      <PositionsTable positions={positions} onSelect={(s) => onSelectSymbol?.(s, mode)}
                      onForceExit={(s) => forceExit(s, mode).then(load)} />
    </>
  );
}

function SessionAnalysis({ mode, onSelectSymbol }) {
  const [trades, setTrades] = useState(null);

  useEffect(() => { api(`/api/trades?mode=${mode}&limit=500`).then(setTrades).catch(() => setTrades([])); }, [mode]);

  if (trades === null) return <p className="hint">loading...</p>;
  const today = new Date().toISOString().slice(0, 10);
  const todays = trades.filter((t) => (t.exit_ts || "").slice(0, 10) === today);
  const sorted = todays.slice().sort((a, b) => a.exit_ts.localeCompare(b.exit_ts));
  let cum = 0;
  const curve = sorted.map((t) => ({ ts: t.exit_ts, equity: (cum += t.pnl_net) }));
  const netPnl = cum;
  const wins = todays.filter((t) => t.pnl_net > 0).length;

  return (
    <div>
      <p className="hint">Market is closed or this session has stopped - showing today's {mode} results.</p>
      <div className="cards">
        <div className="card"><h3>Trades today</h3><div className="row"><span>{todays.length}</span></div></div>
        <div className="card"><h3>Net P&L</h3><div className="row"><span className={cls(netPnl)}>{fmt(netPnl)}</span></div></div>
        <div className="card"><h3>Win rate</h3><div className="row"><span>{todays.length ? `${((wins / todays.length) * 100).toFixed(0)}%` : "-"}</span></div></div>
      </div>
      {curve.length > 0 && <><h2>Today's Equity Curve</h2><EquityCurve data={curve} /></>}
      <h2>Trades</h2>
      <TradesTable trades={todays} onSelect={(s) => onSelectSymbol?.(s, mode)} />
    </div>
  );
}

function StoppedView({ mode, showAnalysis, onSelectSymbol, onStarted, stockCount, manualStart }) {
  return (
    <>
      {showAnalysis ? <SessionAnalysis mode={mode} onSelectSymbol={onSelectSymbol} />
                    : <StartPanel mode={mode} onStarted={onStarted} stockCount={stockCount} manualStart={manualStart} />}
      <OpenPositions mode={mode} onSelectSymbol={onSelectSymbol} />
    </>
  );
}

export default function TradingSession({ mode, marketStatus, onSelectSymbol }) {
  const [status, setStatus] = useState(null);
  const [strategies, setStrategies] = useState([]);
  const [hasTradesToday, setHasTradesToday] = useState(false);
  const [forceStart, setForceStart] = useState(false);
  const [stockCount, setStockCount] = useState(null);   // null until the saved list has loaded

  useEffect(() => {
    let alive = true;
    const load = () => api("/api/status").then((d) => alive && setStatus(d)).catch(() => {});
    load();
    const id = setInterval(load, 4000);
    return () => { alive = false; clearInterval(id); };
  }, []);

  useEffect(() => {
    api("/api/strategies").then((rows) => setStrategies(rows.map((r) => ({ id: r.id, horizon: r.horizon })))).catch(() => {});
  }, []);

  useEffect(() => {
    let alive = true;
    const today = new Date().toISOString().slice(0, 10);
    const check = () => api(`/api/trades?mode=${mode}&limit=5`).then((rows) => {
      if (alive) setHasTradesToday(rows.some((r) => (r.exit_ts || "").slice(0, 10) === today));
    }).catch(() => {});
    check();
    const id = setInterval(check, 10000);
    return () => { alive = false; clearInterval(id); };
  }, [mode]);

  if (!status || !marketStatus) return <p className="hint">loading...</p>;
  const inst = status.config?.instance;
  const follows = !!inst?.follows;
  const running = status[mode]?.running;
  const showAnalysis = !running && !forceStart && (!marketStatus.is_open || hasTradesToday);

  return (
    <section>
      <h2>{inst?.name ? `${inst.name} - ` : ""}{mode === "live" ? "Live Trading" : "Paper Trading"}</h2>
      {status.config && (
        <p className="hint" style={{ margin: "0 0 12px" }}>
          MACD {status.config.timeframe}
          {status.config.signal_mode === "live"
            ? ` from live ticks (signals do not wait for the candle to close; MACD must be more than max(${status.config.tolerance}, ${status.config.tolerance_atr} x the stock's ATR) beyond the signal line for ${status.config.hold_seconds}s)`
            : status.config.signal_mode === "candle_confirmed"
              ? ` from closed candles: buy when MACD crosses above the signal line and clears max(${status.config.tolerance}, ${status.config.tolerance_atr} x the stock's ATR); sell at the next intersection; ${Math.round((status.config.stop_loss_pct || 0) * 100)}% stop; trades ${status.config.trade_start}-${status.config.square_off}`
              : ""} · capital Rs {fmt(status.config.capital)} · system sizes each trade{" "}
          {Math.round(status.config.min_position_pct * 100)}-{Math.round(status.config.max_position_pct * 100)}% of capital ·
          {status.config.max_open_positions ? `max ${status.config.max_open_positions} open positions (a new buy waits for a stock to exit)` : "no limit on open positions (the free cash decides)"}
          {status.config.whole_index ? ` · the whole ${String(status.config.universe).toUpperCase()} is monitored and traded` : ""}
        </p>
      )}
      {!follows && <WatchlistBox mode={mode} wholeIndex={!!status.config?.whole_index} onChange={(list) => setStockCount(list.length)} />}
      <TrendingBox mode={mode} onSelectSymbol={onSelectSymbol} />
      {!running && (
        <button className="ghost" style={{ marginBottom: 12 }} onClick={() => setForceStart(true)}>
          force start {mode} trading
        </button>
      )}
      {running
        ? <LiveSession mode={mode} strategies={strategies} onSelectSymbol={onSelectSymbol}
                      onStopped={() => setStatus((s) => ({ ...s, [mode]: { ...s[mode], running: false } }))} />
        : <StoppedView mode={mode} showAnalysis={showAnalysis} onSelectSymbol={onSelectSymbol} stockCount={stockCount ?? 1} manualStart={follows}
                       onStarted={() => { setForceStart(false); setStatus((s) => ({ ...s, [mode]: { ...s[mode], running: true } })); }} />}
    </section>
  );
}
