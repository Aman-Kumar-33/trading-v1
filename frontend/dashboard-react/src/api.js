import { toast } from "./toast.js";

export async function api(path, opts) {
  const res = await fetch(path, opts && { method: "POST", headers: { "Content-Type": "application/json" },
                                          body: JSON.stringify(opts) });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

export const fmt = (n, d = 0) => (n == null ? "-" : Number(n).toLocaleString("en-IN", { maximumFractionDigits: d }));
export const pct = (n) => (n == null ? "-" : `${(n * 100).toFixed(1)}%`);
export const cls = (n) => (n == null ? "" : n > 0 ? "pos" : n < 0 ? "neg" : "");

// Queue a market exit for every open position in `symbol`. Works even when no trading process is running -
// the next one to start executes it first thing. Returns true once the request was queued.
export async function forceExit(symbol, mode = "paper") {
  if (!window.confirm(`Force exit ${symbol} at market (${mode})? This closes every open position in it and cannot be undone.`)) return false;
  try {
    const r = await api(`/api/positions/${encodeURIComponent(symbol)}/close`, { mode });
    toast(r.process_running
      ? `${symbol}: exit queued - fills within a few seconds`
      : `${symbol}: exit queued - no trading process is running, it executes as soon as one starts`);
    return true;
  } catch (e) {
    toast(e.message, true);
    return false;
  }
}
