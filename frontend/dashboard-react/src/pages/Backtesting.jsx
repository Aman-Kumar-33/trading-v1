import { useState, useEffect } from "react";
import { api, fmt, pct } from "../api.js";
import { EquityCurve, MonthlyHeatmap, IsVsOosBars, StatTile } from "../components/charts.jsx";

export default function Backtesting() {
  const [runs, setRuns] = useState([]);
  const [selected, setSelected] = useState(null);
  const [detail, setDetail] = useState(null);

  useEffect(() => { api("/api/backtests").then(setRuns).catch(() => {}); }, []);

  function select(idx) {
    const r = runs[idx];
    setSelected(r);
    setDetail(null);
    api(`/api/backtests/${r.kind}/${encodeURIComponent(r.id)}`).then(setDetail).catch(() => {});
  }

  return (
    <section>
      <h2>Backtesting</h2>
      <div className="run-picker">
        <select defaultValue="" onChange={(e) => select(e.target.value)}>
          <option value="" disabled>Choose a backtest run...</option>
          {runs.map((r, i) => (
            <option key={i} value={i}>{r.label} ({r.trades} trades, net {fmt(r.net_pnl)})</option>
          ))}
        </select>
      </div>
      {!selected && <p className="hint">Per-strategy backtests, full-system runs, and walk-forward
        results are all listed above - pick one to see its graphs.</p>}

      {detail && selected.kind !== "walk_forward" && (
        <>
          <div className="cards">
            <StatTile label="Sharpe / Sortino" value={`${fmt(detail.metrics.sharpe, 2)} / ${fmt(detail.metrics.sortino, 2)}`} />
            <StatTile label="CAGR" value={pct(detail.metrics.cagr)} />
            <StatTile label="Alpha / Beta" value={`${pct(detail.metrics.alpha)} / ${fmt(detail.metrics.beta, 2)}`} />
            <StatTile label="Win rate" value={pct(detail.metrics.win_rate)} />
            <StatTile label="Profit factor" value={fmt(detail.metrics.profit_factor, 2)} />
            <StatTile label="Max drawdown" value={fmt(detail.metrics.max_drawdown)} />
            <StatTile label="Net P&L" value={fmt(detail.metrics.net_pnl)} />
          </div>
          <h2>Equity Curve vs NIFTY (indexed)</h2>
          <EquityCurve data={detail.equity_curve} />
          <h2>Monthly P&L</h2>
          <MonthlyHeatmap rows={detail.monthly_pnl} />
        </>
      )}

      {detail && selected.kind === "walk_forward" && (
        <>
          <div className="cards">
            <StatTile label="Walk-forward efficiency" value={fmt(detail.summary.walk_forward_efficiency, 2)} />
            <StatTile label="Combined OOS net P&L" value={fmt(detail.summary.combined_out_of_sample?.net_pnl)} />
            <StatTile label="Combined OOS win rate" value={pct(detail.summary.combined_out_of_sample?.win_rate)} />
          </div>
          <h2>In-sample vs Out-of-sample, per fold</h2>
          <IsVsOosBars folds={detail.folds} />
        </>
      )}
    </section>
  );
}
