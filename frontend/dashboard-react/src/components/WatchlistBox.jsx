import { useEffect, useState } from "react";
import { api } from "../api.js";
import { toast } from "../toast.js";

// The stocks the MACD system is allowed to look at. Only these are checked and traded; every other stock is
// ignored. Saved on the server (data/state/watchlist_<mode>.json) - a running trading process picks the change
// up within a couple of seconds, no restart needed.
export default function WatchlistBox({ mode, onChange, wholeIndex = false }) {
  const [saved, setSaved] = useState(null);     // list currently on the server
  const [text, setText] = useState("");         // what's typed in the box
  const [unknown, setUnknown] = useState([]);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let alive = true;
    api(`/api/watchlist?mode=${mode}`).then((d) => {
      if (!alive) return;
      setSaved(d.symbols);
      setText(d.symbols.join(", "));
      onChange?.(d.symbols);
    }).catch(() => alive && setSaved([]));
    return () => { alive = false; };
  }, [mode]);

  async function save() {
    setBusy(true);
    try {
      const d = await api("/api/watchlist", { mode, symbols: text });
      setSaved(d.symbols);
      setText(d.symbols.join(", "));
      setUnknown(d.unknown || []);
      onChange?.(d.symbols);
      toast(d.unknown?.length
        ? `saved ${d.symbols.length} stocks - not found on NSE and skipped: ${d.unknown.join(", ")}`
        : `saved ${d.symbols.length} stocks${d.process_running ? " - applied to the running system" : ""}`,
        !!d.unknown?.length);
    } catch (e) {
      toast(e.message, true);
    } finally {
      setBusy(false);
    }
  }

  if (saved === null) return null;
  const dirty = text.split(/[\s,;]+/).filter(Boolean).map((s) => s.toUpperCase()).join(",") !== saved.join(",");

  return (
    <div className="card" style={{ marginBottom: 16 }}>
      <h3>Stocks to check with MACD</h3>
      <p className="hint" style={{ margin: "0 0 8px" }}>
        Only these stocks are checked for MACD signals and traded - every other stock is ignored.
        Type NSE symbols separated by commas, spaces or new lines. Leave the box empty to {wholeIndex
          ? "trade the whole index - every stock of the NIFTY 50 is monitored (shown below)."
          : "let the system pick today's top trending stocks automatically (shown below)."}
      </p>
      <textarea rows={3} style={{ width: "100%", boxSizing: "border-box" }} value={text}
                placeholder="e.g. RELIANCE, HDFCBANK, ICICIBANK, TCS, INFY"
                onChange={(e) => setText(e.target.value)} />
      <div className="controls" style={{ marginTop: 8 }}>
        <button disabled={busy || !dirty} onClick={save}>{busy ? "saving..." : "Save stock list"}</button>
        <span className="hint">
          {saved.length === 0 ? (wholeIndex ? "no stocks saved - the whole index is traded" : "no stocks saved - auto mode: the system picks today's trending stocks") : `${saved.length} saved${dirty ? " - unsaved changes in the box" : ""}`}
        </span>
      </div>
      {unknown.length > 0 && <p className="neg" style={{ margin: "6px 0 0", fontSize: 12 }}>Not found on NSE (not saved): {unknown.join(", ")}</p>}
    </div>
  );
}
