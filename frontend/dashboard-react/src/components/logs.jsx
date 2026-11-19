import { useMemo } from "react";

const LEVEL_COLORS = { INFO: "var(--accent)", WARNING: "var(--yellow)", ERROR: "var(--red)", CRITICAL: "var(--red)" };
const LEVELS = ["INFO", "WARNING", "ERROR"];

export function summarize(e) {
  const d = e.data || {};
  switch (e.msg) {
    case "position opened": return `${d.strategy || "?"} opened ${d.side || ""} ${d.symbol || ""} x${d.qty || ""} @ ${d.entry ?? ""}`;
    case "position closed": return `${d.strategy || "?"} closed ${d.symbol || ""} (${d.exit_reason || "?"}) pnl ${d.pnl_net ?? ""}`;
    case "signal rejected by risk manager": return `${d.strategy || "?"} rejected on ${d.symbol || ""} - ${d.reason || ""}`;
    case "signal approved by risk manager": return `${d.strategy || "?"} approved on ${d.symbol || ""} qty ${d.qty ?? ""} (weight ${d.weight ?? ""})`;
    case "signals merged": return `merged ${(d.strategies || []).join("+")} on ${d.symbol || ""} (weight ${d.weight ?? ""})`;
    case "signal dropped by allocator": return `allocator dropped ${(d.strategies || []).join("+")} on ${d.symbol || ""} - ${d.reason || ""}`;
    case "shadow trade scored": return `${d.strategy || "?"} shadow-scored ${d.symbol || ""} r=${d.r_net ?? ""}`;
    default: {
      const kv = Object.entries(d).slice(0, 4).map(([k, v]) => `${k}=${v}`).join(" ");
      return kv ? `${e.msg} (${kv})` : e.msg;
    }
  }
}

export function LogEntry({ e }) {
  if (e.raw !== undefined) return <div className="log-entry">{e.raw}</div>;
  return (
    <div className={`log-entry level-${e.level}`}>
      <span className={`dot level-${e.level}`} />
      <span className="time">{(e.ts || "").slice(11, 19) || e.ts}</span>
      <span className="text">{summarize(e)}</span>
    </div>
  );
}

export function LogColumn({ title, entries }) {
  return (
    <div className="log-col">
      <h3>{title} <span className="count-badge">{entries.length}</span></h3>
      <div className="log-list">
        {entries.length === 0 ? <div className="log-empty">nothing here</div>
          : entries.slice().reverse().map((e, i) => <LogEntry key={i} e={e} />)}
      </div>
    </div>
  );
}

function isoWeekLabel(d) {
  const date = new Date(Date.UTC(d.getFullYear(), d.getMonth(), d.getDate()));
  const dayNum = (date.getUTCDay() + 6) % 7;
  date.setUTCDate(date.getUTCDate() - dayNum + 3);
  const firstThursday = new Date(Date.UTC(date.getUTCFullYear(), 0, 4));
  const week = 1 + Math.round(((date - firstThursday) / 86400000 - 3 + ((firstThursday.getUTCDay() + 6) % 7)) / 7);
  return `${date.getUTCFullYear()}-W${String(week).padStart(2, "0")}`;
}

export function bucketKey(e, dim) {
  if (dim === "day" || dim === "week") {
    if (!e.ts) return "(unknown)";
    const d = new Date(e.ts);
    if (isNaN(d)) return "(unknown)";
    return dim === "day" ? e.ts.slice(0, 10) : isoWeekLabel(d);
  }
  if (dim === "stock") return (e.data && e.data.symbol) || "(no symbol)";
  return (e.data && e.data.strategy) || "(no strategy)";
}

export function LogChart({ entries, dim, activeFilter, onBarClick }) {
  const { buckets, keys } = useMemo(() => {
    const buckets = new Map();
    for (const e of entries) {
      if (e.raw !== undefined) continue;
      const key = bucketKey(e, dim);
      if (!buckets.has(key)) buckets.set(key, { INFO: 0, WARNING: 0, ERROR: 0, total: 0 });
      const b = buckets.get(key);
      const lvl = e.level === "CRITICAL" ? "ERROR" : (LEVELS.includes(e.level) ? e.level : "INFO");
      b[lvl]++; b.total++;
    }
    let keys = [...buckets.keys()];
    const chronological = dim === "day" || dim === "week";
    keys.sort(chronological ? (a, b) => a.localeCompare(b) : (a, b) => buckets.get(b).total - buckets.get(a).total);
    keys = chronological ? keys.slice(-21) : keys.slice(0, 15);
    return { buckets, keys };
  }, [entries, dim]);

  if (!keys.length) return <div className="chart-area"><div className="log-empty">no data for this log</div></div>;
  const max = Math.max(...keys.map((k) => buckets.get(k).total), 1);
  const barW = 36, gap = 14, plotH = 200, labelH = 40;
  const width = keys.length * (barW + gap);

  return (
    <div className="chart-area">
      <svg viewBox={`0 0 ${width} ${plotH + labelH}`} width={width} height={plotH + labelH}>
        {keys.map((k, i) => {
          const b = buckets.get(k);
          const x = i * (barW + gap);
          let y = plotH;
          const segs = LEVELS.map((lvl) => {
            const hgt = max ? (b[lvl] / max) * (plotH - 16) : 0;
            if (hgt <= 0) return null;
            y -= hgt;
            return <rect key={lvl} x={x} y={y + 1} width={barW} height={Math.max(hgt - 2, 0)} rx="2" fill={LEVEL_COLORS[lvl]}>
              <title>{`${k}: ${lvl.toLowerCase()} ${b[lvl]}`}</title>
            </rect>;
          });
          const label = k.length > 9 ? k.slice(-9) : k;
          const selected = activeFilter && activeFilter.dim === dim && activeFilter.key === k;
          return (
            <g key={k} className="chart-bar" opacity={activeFilter && !selected ? 0.4 : 1}
               onClick={() => onBarClick(k)}>
              <rect className="hit" x={x} y={0} width={barW} height={plotH} fill="transparent">
                <title>{`${k}: ${b.total} total`}</title>
              </rect>
              {segs}
              <text x={x + barW / 2} y={plotH + 14} textAnchor="middle" className="chart-label">{label}</text>
              <text x={x + barW / 2} y={plotH - (b.total / max) * (plotH - 16) - 4} textAnchor="middle" className="chart-count">{b.total}</text>
            </g>
          );
        })}
      </svg>
      <div className="chart-legend">
        <span className="legend-item"><span className="dot level-INFO" />info</span>
        <span className="legend-item"><span className="dot level-WARNING" />warning</span>
        <span className="legend-item"><span className="dot level-ERROR" />error</span>
      </div>
    </div>
  );
}
