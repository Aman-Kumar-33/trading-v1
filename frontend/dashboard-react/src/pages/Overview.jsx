import { useEffect, useState } from "react";
import { api, fmt, pct, cls, forceExit } from "../api.js";
import { DailyPnlChart, StatTile } from "../components/charts.jsx";
import { TradesTable, PositionsTable, StrategyStatusStrip } from "../components/blotter.jsx";

function pctOf(net, capital) {
  return capital ? `${((net / capital) * 100).toFixed(2)}%` : "-";
}

function MetricCard({ title, m, capital }) {
  if (!m) return null;
  return (
    <div className="card">
      <h3>{title}</h3>
      <div className="row"><span>Trades</span><span>{m.trades}</span></div>
      <div className="row"><span>Gross P&L</span><span className={cls(m.gross_pnl)}>{fmt(m.gross_pnl)}</span></div>
      <div className="row"><span>Costs</span><span>{fmt(m.total_costs)}</span></div>
      <div className="row"><span>Net P&L</span>
        <span className={cls(m.net_pnl)}>{fmt(m.net_pnl)} ({pctOf(m.net_pnl, capital)})</span></div>
      <div className="row"><span>Win rate</span><span>{pct(m.win_rate)}</span></div>
      <div className="row"><span>Profit factor</span><span>{fmt(m.profit_factor, 2)}</span></div>
      <div className="row"><span>Expectancy</span><span className={cls(m.expectancy)}>{fmt(m.expectancy, 1)}</span></div>
      <div className="row"><span>Max drawdown</span><span className="neg">{fmt(m.max_drawdown)}</span></div>
      <div className="row"><span>Sharpe / Sortino</span><span>{fmt(m.sharpe, 2)} / {fmt(m.sortino, 2)}</span></div>
      <div className="row"><span>CAGR</span><span>{pct(m.cagr)}</span></div>
      <div className="row"><span>Alpha / Beta</span><span>{pct(m.alpha)} / {fmt(m.beta, 2)}</span></div>
      {capital != null && <div className="row"><span>Capital</span><span>{fmt(capital)}</span></div>}
    </div>
  );
}

function TodayRow({ paper, live, paperCapital, liveCapital, paperOnly }) {
  return (
    <div className="cards" style={{ marginBottom: 20 }}>
      <StatTile label="Paper - today's net P&L" value={
        <span className={cls(paper.net_pnl)}>{fmt(paper.net_pnl)} ({pctOf(paper.net_pnl, paperCapital)})</span>} />
      <StatTile label="Paper - today's trades / win rate" value={`${paper.trades} / ${pct(paper.win_rate)}`} />
      {!paperOnly && <StatTile label="Live - today's net P&L" value={
        <span className={cls(live.net_pnl)}>{fmt(live.net_pnl)} ({pctOf(live.net_pnl, liveCapital)})</span>} />}
      {!paperOnly && <StatTile label="Live - today's trades / win rate" value={`${live.trades} / ${pct(live.win_rate)}`} />}
    </div>
  );
}

export default function Overview({ overview: data, onSelectSymbol, paperOnly = false }) {
  const [dailyPaper, setDailyPaper] = useState(null);
  const [dailyLive, setDailyLive] = useState(null);
  const [strategyRows, setStrategyRows] = useState([]);
  const [weights, setWeights] = useState({});
  const [todaysTrades, setTodaysTrades] = useState([]);
  const [positions, setPositions] = useState([]);

  useEffect(() => {
    function load() {
      api("/api/daily_pnl?mode=paper&days=30").then(setDailyPaper).catch(() => setDailyPaper([]));
      if (!paperOnly) api("/api/daily_pnl?mode=live&days=30").then(setDailyLive).catch(() => setDailyLive([]));
      api("/api/strategies").then(setStrategyRows).catch(() => {});
      api("/api/ensemble?mode=paper").then(setWeights).catch(() => {});
      api("/api/positions?mode=paper").then(setPositions).catch(() => {});
      const today = new Date().toISOString().slice(0, 10);
      api("/api/trades?mode=paper&limit=200").then((rows) =>
        setTodaysTrades(rows.filter((t) => (t.exit_ts || "").slice(0, 10) === today))).catch(() => {});
    }
    load();
    const id = setInterval(load, 8000);
    return () => clearInterval(id);
  }, []);

  if (!data) return <p className="hint">loading...</p>;

  return (
    <section>
      <h2>Overview</h2>
      <p className="section-label">Today (resets at midnight IST - not the all-time totals below)</p>
      <TodayRow paper={data.paper_today} live={data.live_today}
               paperCapital={data.capital.paper_capital} liveCapital={data.capital.capital} paperOnly={paperOnly} />

      <p className="section-label">Strategy status (paper ensemble weight)</p>
      <StrategyStatusStrip rows={strategyRows} weights={weights} />

      <p className="section-label">Today's trades (paper)</p>
      <TradesTable trades={todaysTrades} onSelect={onSelectSymbol} />

      <p className="section-label">Open positions ({positions.length}) - click a row for the chart</p>
      <PositionsTable positions={positions} onSelect={onSelectSymbol}
                      onForceExit={(s) => forceExit(s, "paper").then(() => api("/api/positions?mode=paper").then(setPositions))} />

      <p className="section-label">Daily P&L - Paper (last 30 trading days)</p>
      {dailyPaper && <DailyPnlChart rows={dailyPaper} />}
      {!paperOnly && <p className="section-label">Daily P&L - Live (last 30 trading days)</p>}
      {!paperOnly && dailyLive && <DailyPnlChart rows={dailyLive} />}

      <p className="section-label">All-time</p>
      <div className="cards">
        <MetricCard title="Paper Trading" m={data.paper} capital={data.capital.paper_capital} />
        {!paperOnly && <MetricCard title="Live Trading" m={data.live} capital={data.capital.capital} />}
      </div>
      <p className="hint">Gross = before transaction/hidden costs. Net = after them. Capital split:
        {" "}{pct(data.capital.intraday_pct)} intraday / {pct(data.capital.long_term_pct)} long-term.</p>
    </section>
  );
}
