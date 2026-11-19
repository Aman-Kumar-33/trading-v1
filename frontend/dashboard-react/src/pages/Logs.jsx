import { useEffect, useState } from "react";
import { api } from "../api.js";
import { LogChart, LogColumn, bucketKey } from "../components/logs.jsx";

const SOURCES = [
  ["run_trading_paper", "paper trading"],
  ["run_trading_live", "live trading"],
  ["backtest_strategies", "backtest"],
  ["backtest_full_system", "full-system backtest"],
  ["walk_forward_strategies", "walk-forward"],
];
const DIMS = ["day", "week", "stock", "strategy"];

export default function Logs() {
  const [source, setSource] = useState(SOURCES[0][0]);
  const [entries, setEntries] = useState([]);
  const [dim, setDim] = useState("day");
  const [filter, setFilter] = useState(null);

  function load() {
    api(`/api/logs?source=${encodeURIComponent(source)}&lines=3000`).then(setEntries).catch(() => {});
  }
  useEffect(() => {
    load();
    setFilter(null);
    if (!source.startsWith("run_trading_")) return;   // only poll live/paper logs - backtest logs are static
    const id = setInterval(load, 8000);
    return () => clearInterval(id);
  }, [source]);

  const filtered = filter ? entries.filter((e) => e.raw === undefined && bucketKey(e, filter.dim) === filter.key) : entries;

  return (
    <section>
      <h2>Logs</h2>
      <div className="controls">
        <select value={source} onChange={(e) => setSource(e.target.value)}>
          {SOURCES.map(([v, label]) => <option key={v} value={v}>{label}</option>)}
        </select>
        <button onClick={load}>Refresh</button>
      </div>
      <div className="controls">
        <div className="seg-group">
          {DIMS.map((d) => (
            <button key={d} className={`seg-btn ${dim === d ? "active" : ""}`} onClick={() => { setDim(d); setFilter(null); }}>
              {d[0].toUpperCase() + d.slice(1)}
            </button>
          ))}
        </div>
      </div>
      <p className="hint">Click a bar to show only that bucket's log lines below.</p>
      <LogChart entries={entries} dim={dim} activeFilter={filter}
                onBarClick={(key) => setFilter((f) => (f && f.key === key && f.dim === dim) ? null : { dim, key })} />
      {filter && (
        <div className="filter-badge">
          Showing {filter.dim}: {filter.key}
          <button onClick={() => setFilter(null)}>&times; clear</button>
        </div>
      )}
      <div className="log-columns">
        <LogColumn title="Intraday" entries={filtered.filter((e) => e.horizon === "INTRADAY")} />
        <LogColumn title="Long-term" entries={filtered.filter((e) => e.horizon === "LONG_TERM")} />
        <LogColumn title="System" entries={filtered.filter((e) => !e.horizon)} />
      </div>
    </section>
  );
}
