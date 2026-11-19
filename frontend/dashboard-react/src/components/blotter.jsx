import { fmt, cls } from "../api.js";

// "2026-11-17T09:44:57+05:30" -> "2026-11-17 09:44:57"
const stamp = (ts) => (ts ? String(ts).slice(0, 19).replace("T", " ") : "");

export function PositionsTable({ positions, onSelect, onForceExit }) {
  return (
    <table>
      <thead><tr><th>Symbol</th><th>Strategy</th><th>Side</th><th>Qty</th><th>Entry</th><th>LTP</th><th>Unrealized P&L</th>
        <th>Stop</th><th>Opened</th><th>Carried</th>{onForceExit && <th></th>}</tr></thead>
      <tbody>
        {positions.map((p, i) => (
          <tr key={i} className={onSelect ? "clickable-row" : ""} onClick={() => onSelect?.(p.symbol)}>
            <td>{p.symbol}</td><td>{p.strategy_id}</td><td>{p.side}</td><td>{p.qty}</td><td>{fmt(p.entry_price, 2)}</td>
            <td>{p.ltp != null ? fmt(p.ltp, 2) : "-"}</td>
            <td className={cls(p.unrealized_pnl)}>{p.unrealized_pnl != null ? fmt(p.unrealized_pnl) : "-"}</td>
            <td>{fmt(p.stop_loss, 2)}</td><td className="mono">{stamp(p.opened_at)}</td><td>{p.carried ? "yes" : ""}</td>
            {onForceExit && (
              <td><button className="danger" onClick={(e) => { e.stopPropagation(); onForceExit(p.symbol); }}>force exit</button></td>
            )}</tr>
        ))}
        {positions.length === 0 && <tr><td colSpan={onForceExit ? 11 : 10} className="hint">no open positions</td></tr>}
      </tbody>
    </table>
  );
}

export function TradesTable({ trades, onSelect }) {
  return (
    <table>
      <thead><tr><th>Symbol</th><th>Strategy</th><th>Side</th><th>Entry (date, time)</th><th>Exit (date, time)</th>
        <th>Qty</th><th>Reason</th><th>Net P&L</th></tr></thead>
      <tbody>
        {trades.map((t, i) => (
          <tr key={i} className={onSelect ? "clickable-row" : ""} onClick={() => onSelect?.(t.symbol)}>
            <td>{t.symbol}</td><td>{t.strategy_id}</td><td>{t.side}</td>
            <td className="mono">{stamp(t.entry_ts)}</td><td className="mono">{stamp(t.exit_ts)}</td>
            <td>{t.qty}</td><td>{t.exit_reason}</td>
            <td className={cls(t.pnl_net)}>{fmt(t.pnl_net)}</td></tr>
        ))}
        {trades.length === 0 && <tr><td colSpan="8" className="hint">no trades yet today</td></tr>}
      </tbody>
    </table>
  );
}

export function StrategyStatusStrip({ rows, weights }) {
  const promoted = rows.filter((r) => r.status === "PAPER" || r.status === "LIVE");
  return (
    <div className="flow-nodes" style={{ marginBottom: 16 }}>
      {promoted.map((r) => {
        const w = weights[r.id] || {};
        const maxW = Math.max(0, ...Object.values(w).map((v) => v.weight ?? 0));
        const muted = maxW <= 0.05;
        return (
          <div key={r.id} className={`flow-node ${muted ? "" : "pulse-good"}`} title={`max weight ${maxW.toFixed(2)}`}>
            {r.id} - {muted ? "muted" : "active"}
          </div>
        );
      })}
    </div>
  );
}
