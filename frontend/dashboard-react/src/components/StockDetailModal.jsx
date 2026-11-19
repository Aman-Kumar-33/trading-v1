import { useEffect, useState } from "react";
import { api, fmt, forceExit as queueForceExit } from "../api.js";
import StockChart from "./StockChart.jsx";

const TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "1d"];

export default function StockDetailModal({ symbol, mode = "paper", onClose }) {
  const [timeframe, setTimeframe] = useState("15m");
  const [data, setData] = useState(null);
  const [openPositions, setOpenPositions] = useState([]);
  const [exitQueued, setExitQueued] = useState(false);

  useEffect(() => {
    if (!symbol) return;
    let alive = true;
    let timer = null;
    let tries = 0;
    setData(null);
    const empty = { candles: [], macd: [], signals: [], trades: [] };
    const lastTs = (d) => (d?.candles?.length ? d.candles[d.candles.length - 1].ts : "");
    // the server answers at once from what it has; if that is behind the clock it fetches the missing bars in the
    // background (`stale`), so ask again shortly - then keep the chart growing while it is open
    const load = () => api(`/api/candles?symbol=${encodeURIComponent(symbol)}&timeframe=${timeframe}&mode=${mode}&limit=300`)
      .then((d) => {
        if (!alive) return;
        tries += 1;
        setData((prev) => (prev && !d.stale && lastTs(prev) === lastTs(d) && prev.candles.length === d.candles.length ? prev : d));
        timer = setTimeout(load, d.stale && tries < 12 ? 3000 : 20000);
      })
      .catch(() => { if (!alive) return; setData((prev) => prev || empty); timer = setTimeout(load, 20000); });
    load();
    return () => { alive = false; clearTimeout(timer); };
  }, [symbol, timeframe, mode]);

  useEffect(() => {
    if (!symbol) return;
    let alive = true;
    setExitQueued(false);
    const load = () => api(`/api/positions?mode=${mode}`)
      .then((rows) => alive && setOpenPositions((rows || []).filter((p) => p.symbol === symbol)))
      .catch(() => alive && setOpenPositions([]));
    load();
    const id = setInterval(load, 3000);
    return () => { alive = false; clearInterval(id); };
  }, [symbol, mode]);

  if (!symbol) return null;

  async function forceExit() {
    if (await queueForceExit(symbol, mode)) setExitQueued(true);
  }

  const recentSignals = (data?.signals || []).slice(-12).reverse();
  // a candle is labelled with its START; the system acts when it CLOSES (start + timeframe) - that is the time of the order
  const tfMin = { "1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60 }[timeframe];
  const orderAt = (ts) => {
    if (!tfMin || !ts) return "-";
    const m = parseInt(ts.slice(11, 13), 10) * 60 + parseInt(ts.slice(14, 16), 10) + tfMin;
    return `${String(Math.floor(m / 60)).padStart(2, "0")}:${String(m % 60).padStart(2, "0")}`;
  };
  const last = data?.macd?.length ? data.macd[data.macd.length - 1] : null;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-panel" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>{symbol}</h2>
          <div className="seg-group">
            {TIMEFRAMES.map((tf) => (
              <button key={tf} className={`seg-btn ${timeframe === tf ? "active" : ""}`} onClick={() => setTimeframe(tf)}>{tf}</button>
            ))}
          </div>
          {openPositions.length > 0 && (
            <button className="danger" disabled={exitQueued} onClick={forceExit}>
              {exitQueued ? "exit queued" : `force exit (${openPositions.map((p) => `${p.qty} @ ${fmt(p.entry_price, 2)}`).join(", ")})`}
            </button>
          )}
          <button className="ghost" onClick={onClose}>close</button>
        </div>
        {exitQueued && <p className="hint">exit queued - the table above refreshes once the trading process has filled it.</p>}
        {data?.context && (
          <p className="hint" style={{ margin: "0 0 8px" }}>
            {data.context.signal_mode === "candle_confirmed"
              ? "BUY arrow = MACD crossed above its signal line and cleared the tolerance on a closed candle (the bar this system buys on); SELL arrow = the next cross down. "
              : "Arrows mark every MACD cross on a closed candle. "}
            {!data.context.running && <b>The {data.context.system} system is not running, so nothing is bought or sold - press Start Paper Trading on its page. </b>}
            {data.context.running && !data.context.listed && <b>{symbol} is not on today's stock list, so it is never bought. </b>}
            {data.context.trade_start ? `Buys only between ${data.context.trade_start} and ${data.context.square_off}; everything is sold at ${data.context.square_off}, so cross-down arrows after it are not drawn - the OUT marker is the exit.` : ""}
          </p>
        )}
        {!data && <p className="hint">loading...</p>}
        {data && data.candles.length === 0 && <p className="hint">no candles for {symbol} on {timeframe} yet.</p>}
        {data && data.candles.length > 0 && (
          <StockChart candles={data.candles} macd={data.macd} signals={data.signals} trades={data.trades} timeframe={timeframe} />
        )}
        {last && (
          <p className="hint">latest bar {last.ts.slice(0, 16)} - MACD {last.macd} / signal {last.signal} /
            histogram {last.hist} ({last.macd > last.signal ? "MACD above signal" : "MACD below signal"})</p>
        )}
        {recentSignals.length > 0 && (
          <>
            <p className="section-label">MACD crosses on this chart (newest first) - compare with your own chart</p>
            <table>
              <thead><tr><th>Candle (starts)</th><th>Order time (candle closes)</th><th>Cross</th><th>Bar close</th><th>MACD</th><th>Signal</th></tr></thead>
              <tbody>
                {recentSignals.map((s, i) => (
                  <tr key={i}><td>{s.ts.slice(0, 16)}</td><td>{s.ts.slice(0, 10)} {orderAt(s.ts)}</td>
                    <td className={s.side === "BUY" ? "pos" : "neg"}>{s.side === "BUY" ? "bullish - BUY" : "bearish - SELL"}</td>
                    <td>{fmt(s.close, 2)}</td><td>{s.macd}</td><td>{s.signal}</td></tr>
                ))}
              </tbody>
            </table>
          </>
        )}
        {data && data.trades && data.trades.length > 0 && (
          <>
            <p className="section-label">Trades taken in {symbol}</p>
            <table>
              <thead><tr><th>Strategy</th><th>Side</th><th>Qty</th><th>Entry</th><th>Exit</th><th>Reason</th><th>Net P&L</th></tr></thead>
              <tbody>
                {data.trades.map((t, i) => (
                  <tr key={i}><td>{t.strategy_id}</td><td>{t.side}</td><td>{t.qty}</td>
                    <td>{(t.entry_ts || "").slice(0, 16).replace("T", " ")} @ {t.entry_price}</td>
                    <td>{(t.exit_ts || "").slice(0, 16).replace("T", " ")} @ {t.exit_price}</td>
                    <td>{t.exit_reason}</td><td className={t.pnl_net >= 0 ? "pos" : "neg"}>{t.pnl_net?.toFixed(0)}</td></tr>
                ))}
              </tbody>
            </table>
          </>
        )}
      </div>
    </div>
  );
}
