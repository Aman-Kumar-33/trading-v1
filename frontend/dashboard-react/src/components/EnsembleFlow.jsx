import { useEffect, useRef, useState } from "react";

// Parent passes only the NEWLY fetched events each poll (via /api/events?since=) - so every item here
// is guaranteed not-yet-animated, no dedup bookkeeping needed.
function eventPulses(e) {
  const d = e.data || {};
  const out = [];
  switch (e.msg) {
    case "signal approved by risk manager":
      out.push(["strategy:" + d.strategy, "pulse-info"], ["stage:allocator", "pulse-info"],
               ["arrow:0", "active"], ["stage:risk", "pulse-good"]);
      break;
    case "signal rejected by risk manager":
      out.push(["strategy:" + d.strategy, "pulse-bad"], ["arrow:0", "active"], ["stage:risk", "pulse-bad"]);
      break;
    case "signals merged":
      (d.strategies || []).forEach((s) => out.push(["strategy:" + s, "pulse-info"]));
      out.push(["stage:allocator", "pulse-info"]);
      break;
    case "signal dropped by allocator":
      (d.strategies || []).forEach((s) => out.push(["strategy:" + s, "pulse-bad"]));
      out.push(["stage:allocator", "pulse-bad"]);
      break;
    case "position opened":
      out.push(["arrow:1", "active"], ["stage:positions", "pulse-good"], ["strategy:" + d.strategy, "pulse-good"]);
      break;
    case "position closed": {
      const good = (d.pnl_net ?? 0) >= 0;
      out.push(["arrow:1", "active"], ["stage:positions", good ? "pulse-good" : "pulse-bad"],
               ["strategy:" + d.strategy, good ? "pulse-good" : "pulse-bad"]);
      break;
    }
    case "shadow trade scored":
      out.push(["strategy:" + d.strategy, "pulse-info"]);
      break;
    default:
      break;
  }
  return out;
}

export default function EnsembleFlow({ strategies, events }) {
  const [pulses, setPulses] = useState({});
  const timers = useRef({});

  useEffect(() => {
    if (!events || !events.length) return;
    for (const e of events) {
      for (const [key, pulseClass] of eventPulses(e)) {
        setPulses((p) => ({ ...p, [key]: pulseClass }));
        clearTimeout(timers.current[key]);
        timers.current[key] = setTimeout(() => {
          setPulses((p) => { const n = { ...p }; delete n[key]; return n; });
        }, 900);
      }
    }
  }, [events]);

  useEffect(() => () => Object.values(timers.current).forEach(clearTimeout), []);

  const intraday = strategies.filter((s) => s.horizon === "INTRADAY");
  const longTerm = strategies.filter((s) => s.horizon === "LONG_TERM");

  return (
    <div className="ensemble-flow">
      <div className="flow-row">
        <div className="row-label">Intraday</div>
        <div className="flow-nodes">
          {intraday.map((s) => <div key={s.id} className={`flow-node ${pulses["strategy:" + s.id] || ""}`}>{s.id}</div>)}
        </div>
      </div>
      <div className="flow-row">
        <div className="row-label">Long-term</div>
        <div className="flow-nodes">
          {longTerm.map((s) => <div key={s.id} className={`flow-node ${pulses["strategy:" + s.id] || ""}`}>{s.id}</div>)}
        </div>
      </div>
      <div className="flow-row">
        <div className="row-label">Pipeline</div>
        <div className="flow-nodes" style={{ alignItems: "center" }}>
          <div className={`flow-stage ${pulses["stage:allocator"] || ""}`}>Allocator</div>
          <span className={`flow-arrow ${pulses["arrow:0"] || ""}`}>&rarr;{pulses["arrow:0"] && <span className="flow-dot" />}</span>
          <div className={`flow-stage ${pulses["stage:risk"] || ""}`}>Risk Manager</div>
          <span className={`flow-arrow ${pulses["arrow:1"] || ""}`}>&rarr;{pulses["arrow:1"] && <span className="flow-dot" />}</span>
          <div className={`flow-stage ${pulses["stage:positions"] || ""}`}>Positions</div>
        </div>
      </div>
    </div>
  );
}
