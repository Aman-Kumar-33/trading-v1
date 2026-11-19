import { useEffect, useState } from "react";
import { api, fmt, pct, cls } from "../api.js";
import { EquityCurve, StatTile } from "../components/charts.jsx";

function Cell({ m, field, isPct, decimals }) {
  if (!m) return <td>-</td>;
  const v = m[field];
  return <td className={cls(v)}>{isPct ? pct(v) : fmt(v, decimals)}</td>;
}

export default function Strategies() {
  const [rows, setRows] = useState([]);
  const [selected, setSelected] = useState(null);
  const [detail, setDetail] = useState(null);
  const [liveCurve, setLiveCurve] = useState(null);
  const [weights, setWeights] = useState(null);

  useEffect(() => {
    let alive = true;
    const load = () => api("/api/strategies").then((d) => alive && setRows(d)).catch(() => {});
    load();
    const id = setInterval(load, 5000);
    return () => { alive = false; clearInterval(id); };
  }, []);

  function select(sid) {
    setSelected(sid);
    setDetail(null);
    setLiveCurve(null);
    setWeights(null);
    api(`/api/backtests/strategy/${encodeURIComponent(sid)}`).then(setDetail).catch(() => setDetail({ error: "no backtest result on disk for this strategy yet" }));
    api(`/api/strategy_equity?id=${encodeURIComponent(sid)}&mode=paper`).then(setLiveCurve).catch(() => setLiveCurve([]));
    api("/api/ensemble?mode=paper").then((w) => setWeights(w[sid] || {})).catch(() => setWeights({}));
  }

  return (
    <section>
      <h2>Strategy Vault</h2>
      <table className="vault-table">
        <thead>
          <tr><th rowSpan="2">Strategy</th><th rowSpan="2">Horizon</th><th rowSpan="2">Status</th>
            <th colSpan="4">Backtest</th><th colSpan="3">Paper</th><th colSpan="3">Live</th></tr>
          <tr><th>Gross</th><th>Net</th><th>Win%</th><th>Sharpe</th>
            <th>Gross</th><th>Net</th><th>Win%</th>
            <th>Gross</th><th>Net</th><th>Win%</th></tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.id} onClick={() => select(r.id)}>
              <td>{r.id}</td><td>{r.horizon}</td><td className={`badge-${r.status}`}>{r.status}</td>
              <Cell m={r.backtest} field="gross_pnl" /><Cell m={r.backtest} field="net_pnl" />
              <Cell m={r.backtest} field="win_rate" isPct /><Cell m={r.backtest} field="sharpe" decimals={2} />
              <Cell m={r.paper} field="gross_pnl" /><Cell m={r.paper} field="net_pnl" /><Cell m={r.paper} field="win_rate" isPct />
              <Cell m={r.live} field="gross_pnl" /><Cell m={r.live} field="net_pnl" /><Cell m={r.live} field="win_rate" isPct />
            </tr>
          ))}
        </tbody>
      </table>
      {selected && (
        <div style={{ marginTop: 20 }}>
          <h2>Inspector: {selected}</h2>
          <p className="section-label">Backtest equity curve</p>
          {detail && detail.error && <p className="hint">{detail.error}</p>}
          {detail && !detail.error && <EquityCurve data={detail.equity_curve} />}
          <p className="section-label">Live strategy performance (paper)</p>
          {liveCurve && <EquityCurve data={liveCurve} />}
          {weights && Object.keys(weights).length > 0 && (
            <div className="cards">
              {Object.entries(weights).map(([regime, w]) => (
                <StatTile key={regime} label={regime} value={`n=${w.n} mean R=${fmt(w.mean_r, 3)}`} />
              ))}
            </div>
          )}
        </div>
      )}
    </section>
  );
}
