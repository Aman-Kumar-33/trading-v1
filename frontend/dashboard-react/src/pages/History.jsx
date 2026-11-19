import { useEffect, useMemo, useState } from "react";
import { api, fmt, cls } from "../api.js";
import { TradesTable } from "../components/blotter.jsx";

export default function History({ onSelectSymbol }) {
  const [mode, setMode] = useState("paper");
  const [trades, setTrades] = useState(null);
  const [day, setDay] = useState(null);

  useEffect(() => {
    setTrades(null);
    setDay(null);
    api(`/api/trades?mode=${mode}&limit=3000`).then(setTrades).catch(() => setTrades([]));
  }, [mode]);

  const byDay = useMemo(() => {
    if (!trades) return [];
    const groups = {};
    for (const t of trades) {
      const d = (t.exit_ts || "").slice(0, 10);
      if (!d) continue;
      (groups[d] ||= []).push(t);
    }
    return Object.entries(groups)
      .map(([d, rows]) => ({ date: d, trades: rows, net: rows.reduce((s, r) => s + (r.pnl_net || 0), 0) }))
      .sort((a, b) => b.date.localeCompare(a.date));
  }, [trades]);

  const selectedDay = byDay.find((d) => d.date === day);

  return (
    <section>
      <h2>History</h2>
      <div className="seg-group" style={{ marginBottom: 14 }}>
        <button className={`seg-btn ${mode === "paper" ? "active" : ""}`} onClick={() => setMode("paper")}>Paper</button>
        <button className={`seg-btn ${mode === "live" ? "active" : ""}`} onClick={() => setMode("live")}>Live</button>
      </div>

      {trades === null && <p className="hint">loading...</p>}

      {!day && trades && (
        <table>
          <thead><tr><th>Date</th><th>Trades</th><th>Net P&L</th></tr></thead>
          <tbody>
            {byDay.map((d) => (
              <tr key={d.date} className="clickable-row" onClick={() => setDay(d.date)}>
                <td>{d.date}</td><td>{d.trades.length}</td>
                <td className={cls(d.net)}>{fmt(d.net)}</td></tr>
            ))}
            {byDay.length === 0 && <tr><td colSpan="3" className="hint">no closed trades yet</td></tr>}
          </tbody>
        </table>
      )}

      {day && selectedDay && (
        <>
          <button className="ghost" style={{ marginBottom: 10 }} onClick={() => setDay(null)}>&larr; back to days</button>
          <p className="section-label">{day} - net {fmt(selectedDay.net)}</p>
          <TradesTable trades={selectedDay.trades} onSelect={(s) => onSelectSymbol(s, mode)} />
        </>
      )}
    </section>
  );
}
