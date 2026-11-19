import { useEffect, useState } from "react";
import { api } from "./api.js";
import Overview from "./pages/Overview.jsx";
import Strategies from "./pages/Strategies.jsx";
import TradingSession from "./pages/TradingSession.jsx";
import Backtesting from "./pages/Backtesting.jsx";
import History from "./pages/History.jsx";
import Logs from "./pages/Logs.jsx";
import { CircularGauge } from "./components/charts.jsx";
import StockDetailModal from "./components/StockDetailModal.jsx";

const TABS = [
  ["overview", "Overview"],
  ["history", "History"],
  ["strategies", "Strategy Vault"],
  ["paper", "Paper Trading"],
  ["live", "Live Trading"],
  ["backtesting", "Backtesting"],
  ["logs", "Logs"],
];

// The "5 minute" system is a separate server on its own port with its own database: its own Overview (paper only),
// trading page, history and logs. Nothing of it appears in the main system's Overview.
const FOLLOWER_TABS = [
  ["overview", "Overview"],
  ["paper", "Paper Trading"],
  ["history", "History"],
  ["logs", "Logs"],
];

export default function App() {
  const [tab, setTab] = useState("overview");
  const [config, setConfig] = useState(null);   // /api/status config: timeframe, limits, which system this is
  const inst = config?.instance;
  const isFollower = !!inst?.follows;
  const tabs = isFollower ? FOLLOWER_TABS : TABS;
  const [marketStatus, setMarketStatus] = useState(null);
  const [overview, setOverview] = useState(null);
  const [selected, setSelected] = useState(null);   // { symbol, mode } | null
  const onSelectSymbol = (symbol, selMode = "paper") => setSelected(symbol ? { symbol, mode: selMode } : null);

  useEffect(() => {
    let alive = true;
    const load = () => api("/api/market_status").then((d) => alive && setMarketStatus(d)).catch(() => {});
    load();
    const id = setInterval(load, 15000);
    return () => { alive = false; clearInterval(id); };
  }, []);

  useEffect(() => {
    api("/api/status").then((d) => setConfig(d.config)).catch(() => {});
  }, []);

  useEffect(() => {
    document.title = inst?.name ? `Algo System - ${inst.name}` : "Algo System";
  }, [isFollower, inst?.name]);

  useEffect(() => {
    if (config === null) return undefined;
    let alive = true;
    const load = () => api("/api/overview").then((d) => alive && setOverview(d)).catch(() => {});
    load();
    const id = setInterval(load, 5000);
    return () => { alive = false; clearInterval(id); };
  }, [config === null]);

  return (
    <>
      <header className="topbar">
        <h1>Algo System{inst?.name ? ` - ${inst.name}` : ""}</h1>
        {marketStatus && <span className={`market-badge ${marketStatus.is_open ? "open" : "closed"}`}>{marketStatus.state}</span>}
      </header>
      <div className="shell">
        <aside className="sidebar">
          {tabs.map(([key, label]) => (
            <button key={key} className={`sidebar-btn ${tab === key ? "active" : ""}`} onClick={() => setTab(key)}>{label}</button>
          ))}
          {(inst?.links || []).map((l) => (
            <a key={l.url} className="sidebar-btn" href={l.key ? `/open/${l.key}` : l.url} target="_blank" rel="noreferrer"
               style={{ textDecoration: "none", display: "block" }}>{l.name} &#8599;</a>
          ))}
          {overview && (
            <div className="sidebar-gauges">
              <CircularGauge label="Paper P&L" value={overview.paper.net_pnl} max={Math.max(Math.abs(overview.paper.net_pnl), overview.capital.paper_capital * 0.02, 1)} />
              {!isFollower && <CircularGauge label="Live P&L" value={overview.live.net_pnl} max={Math.max(Math.abs(overview.live.net_pnl), (overview.capital.capital || 1) * 0.02, 1)} />}
            </div>
          )}
        </aside>
        <main className="main-content">
          {tab === "overview" && <Overview overview={overview} onSelectSymbol={onSelectSymbol} paperOnly={isFollower} />}
          {tab === "history" && <History onSelectSymbol={onSelectSymbol} />}
          {tab === "strategies" && <Strategies />}
          {tab === "paper" && <TradingSession mode="paper" marketStatus={marketStatus} onSelectSymbol={onSelectSymbol} />}
          {tab === "live" && !isFollower && <TradingSession mode="live" marketStatus={marketStatus} onSelectSymbol={onSelectSymbol} />}
          {tab === "backtesting" && <Backtesting />}
          {tab === "logs" && <Logs />}
        </main>
      </div>
      <StockDetailModal symbol={selected?.symbol} mode={selected?.mode} onClose={() => setSelected(null)} />
    </>
  );
}
