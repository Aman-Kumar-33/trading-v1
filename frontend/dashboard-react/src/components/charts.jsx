import { fmt } from "../api.js";

// Hand-rolled SVG charts - no charting library, consistent with the rest of this dashboard. Native
// <title> elements give every mark a hover tooltip for free (the dataviz skill's "always ship a hover
// layer" rule) without any JS event wiring.

export function EquityCurve({ data, height = 220, width = 760 }) {
  if (!data || !data.length) return <div className="log-empty">no equity data for this run</div>;
  const hasBench = data.some((d) => d.benchmark_equity != null);
  const hasAlpha = data.some((d) => d.alpha_equity != null);
  const all = data.flatMap((d) => [d.equity, hasBench ? d.benchmark_equity : null]).filter((v) => v != null);
  const min = Math.min(...all), max = Math.max(...all);
  const pad = (max - min) * 0.05 || 1;
  const y = (v) => height - 20 - ((v - min + pad) / (max - min + 2 * pad)) * (height - 40);
  const x = (i) => (i / (data.length - 1 || 1)) * (width - 20) + 10;
  const path = (key) => data.map((d, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(d[key]).toFixed(1)}`).join(" ");

  // alpha is a much smaller series than equity/benchmark (it's an excess-return residual) - its own
  // zero-centered scale, plotted in the bottom third of the same chart rather than a second y-axis
  // (the dataviz skill's "never dual-axis" rule - two charts/zones, not two scales on one).
  const alphaVals = hasAlpha ? data.map((d) => d.alpha_equity) : [];
  const alphaMax = Math.max(...alphaVals.map(Math.abs), 1);
  const alphaBandH = height * 0.28;
  const alphaY = (v) => height - alphaBandH / 2 - (v / alphaMax) * (alphaBandH / 2 - 4);
  const alphaPath = hasAlpha
    ? data.map((d, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${alphaY(d.alpha_equity).toFixed(1)}`).join(" ")
    : "";

  return (
    <div className="chart-area">
      <svg viewBox={`0 0 ${width} ${height}`} width={width} height={height}>
        <defs>
          <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>
        {hasBench && <path d={path("benchmark_equity")} fill="none" stroke="var(--muted)" strokeWidth="1.5" strokeDasharray="4,3" />}
        <path d={path("equity")} fill="none" stroke="var(--accent)" strokeWidth="2" filter="url(#glow)" />
        {hasAlpha && <path d={alphaPath} fill="none" stroke="var(--purple)" strokeWidth="1.5" opacity="0.9" />}
        {data.map((d, i) => (i % Math.ceil(data.length / 40 || 1) === 0 ? (
          <circle key={i} cx={x(i)} cy={y(d.equity)} r="2.5" fill="var(--accent)">
            <title>{`${d.ts.slice(0, 10)}: ${fmt(d.equity)}`}</title>
          </circle>
        ) : null))}
      </svg>
      <div className="chart-legend">
        <span className="legend-item"><span className="dot" style={{ background: "var(--accent)" }} />equity</span>
        {hasBench && <span className="legend-item"><span className="dot" style={{ background: "var(--muted)" }} />NIFTY (indexed)</span>}
        {hasAlpha && <span className="legend-item"><span className="dot" style={{ background: "var(--purple)" }} />cumulative alpha (excess over beta-adjusted benchmark)</span>}
      </div>
    </div>
  );
}

export function CircularGauge({ label, value, max, size = 88 }) {
  const r = size / 2 - 6;
  const circumference = 2 * Math.PI * r;
  const t = max ? Math.max(0, Math.min(1, Math.abs(value) / max)) : 0;
  const color = value >= 0 ? "var(--green)" : "var(--red)";
  return (
    <div className="radial-gauge">
      <div className="radial-gauge-ring" style={{ width: size, height: size }}>
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
          <defs>
            <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
          </defs>
          <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--border)" strokeWidth="6" />
          <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={color} strokeWidth="6" strokeLinecap="round"
                  strokeDasharray={circumference} strokeDashoffset={circumference * (1 - t)}
                  transform={`rotate(-90 ${size / 2} ${size / 2})`} filter="url(#glow)" style={{ transition: "stroke-dashoffset 0.6s" }} />
        </svg>
        <div className="value mono" style={{ color }}>{fmt(value, 0)}</div>
      </div>
      <div className="label">{label}</div>
    </div>
  );
}

export function MonthlyHeatmap({ rows }) {
  if (!rows || !rows.length) return <div className="log-empty">no monthly data</div>;
  const years = [...new Set(rows.map((r) => r.year))].sort();
  const months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
  const byKey = Object.fromEntries(rows.map((r) => [`${r.year}-${r.month}`, r.net_pnl]));
  const max = Math.max(...rows.map((r) => Math.abs(r.net_pnl)), 1);
  const color = (v) => {
    if (v == null) return "var(--panel-2)";
    const t = Math.min(1, Math.abs(v) / max);
    return v >= 0 ? `rgba(46,204,113,${0.15 + 0.6 * t})` : `rgba(255,92,92,${0.15 + 0.6 * t})`;
  };
  return (
    <div className="heatmap" style={{ gridTemplateColumns: `60px repeat(${months.length}, 1fr)` }}>
      <div />
      {months.map((m) => <div key={m} className="cell col-head">{m}</div>)}
      {years.map((y) => (
        <>
          <div key={`y${y}`} className="cell row-head">{y}</div>
          {months.map((m) => {
            const v = byKey[`${y}-${m}`];
            return (
              <div key={`${y}-${m}`} className="cell" style={{ background: color(v) }} title={v != null ? fmt(v) : ""}>
                {v != null ? fmt(v) : ""}
              </div>
            );
          })}
        </>
      ))}
    </div>
  );
}

export function StatTile({ label, value }) {
  return (
    <div className="gauge">
      <div className="label">{label}</div>
      <div className="value mono">{value}</div>
    </div>
  );
}

export function Gauge({ label, value, max, suffix = "" }) {
  const t = max ? Math.max(0, Math.min(1, value / max)) : 0;
  return (
    <div className="gauge">
      <div className="label">{label}</div>
      <div className="value mono">{fmt(value)}{suffix}</div>
      <div className="bar-track"><div className="bar-fill" style={{ width: `${t * 100}%` }} /></div>
    </div>
  );
}

export function DailyPnlChart({ rows, height = 200 }) {
  if (!rows || !rows.length) return <div className="log-empty">no closed trades yet</div>;
  const max = Math.max(...rows.map((r) => Math.abs(r.net)), 1);
  const barW = 26, gap = 14, plotH = height - 34;
  const width = rows.length * (barW + gap) + 20;
  const y0 = plotH / 2;   // zero-line in the middle - P&L is signed, bars grow up (profit) or down (loss)
  const scale = (plotH / 2 - 10) / max;

  return (
    <div className="chart-area">
      <svg viewBox={`0 0 ${width} ${height}`} width={width} height={height}>
        <line x1="10" x2={width - 10} y1={y0} y2={y0} stroke="var(--border)" strokeWidth="1" />
        {rows.map((r, i) => {
          const x = i * (barW + gap) + 15;
          const h = Math.abs(r.net) * scale;
          const good = r.net >= 0;
          const y = good ? y0 - h : y0;
          const color = good ? "var(--green)" : "var(--red)";
          return (
            <g key={r.day}>
              <rect x={x} y={y} width={barW} height={Math.max(h, 1)} rx="2" fill={color}>
                <title>{`${r.day}: ${r.trades} trades, net ${fmt(r.net)} (gross ${fmt(r.gross)}, costs ${fmt(r.costs)})`}</title>
              </rect>
              <text x={x + barW / 2} y={good ? y - 4 : y + h + 12} textAnchor="middle" className="chart-count">
                {fmt(r.net, 0)}
              </text>
              <text x={x + barW / 2} y={height - 6} textAnchor="middle" className="chart-label">
                {r.day.slice(5)}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="chart-legend">
        <span className="legend-item"><span className="dot" style={{ background: "var(--green)" }} />profit day</span>
        <span className="legend-item"><span className="dot" style={{ background: "var(--red)" }} />loss day</span>
      </div>
    </div>
  );
}

export function IsVsOosBars({ folds }) {
  if (!folds || !folds.length) return <div className="log-empty">no walk-forward folds</div>;
  const metric = "net_pnl";
  const max = Math.max(...folds.flatMap((f) => [Math.abs(f.in_sample?.[metric] || 0), Math.abs(f.out_of_sample?.[metric] || 0)]), 1);
  const barW = 14, gap = 26, groupGap = 18, h = 140;
  return (
    <div className="chart-area">
      <svg viewBox={`0 0 ${folds.length * (barW * 2 + gap + groupGap)} ${h + 24}`}
           width={folds.length * (barW * 2 + gap + groupGap)} height={h + 24}>
        {folds.map((f, i) => {
          const x0 = i * (barW * 2 + gap + groupGap);
          const isV = f.in_sample?.[metric] || 0, oosV = f.out_of_sample?.[metric] || 0;
          const hIs = (Math.abs(isV) / max) * (h - 10), hOos = (Math.abs(oosV) / max) * (h - 10);
          return (
            <g key={i}>
              <rect x={x0} y={h - hIs} width={barW} height={hIs} fill="var(--accent)">
                <title>{`fold ${i + 1} in-sample: ${fmt(isV)}`}</title>
              </rect>
              <rect x={x0 + barW + gap / 2} y={h - hOos} width={barW} height={hOos} fill={oosV >= 0 ? "var(--green)" : "var(--red)"}>
                <title>{`fold ${i + 1} out-of-sample: ${fmt(oosV)}`}</title>
              </rect>
              <text x={x0 + barW} y={h + 14} textAnchor="middle" className="chart-label">{`F${i + 1}`}</text>
            </g>
          );
        })}
      </svg>
      <div className="chart-legend">
        <span className="legend-item"><span className="dot" style={{ background: "var(--accent)" }} />in-sample</span>
        <span className="legend-item"><span className="dot" style={{ background: "var(--green)" }} />out-of-sample</span>
      </div>
    </div>
  );
}
