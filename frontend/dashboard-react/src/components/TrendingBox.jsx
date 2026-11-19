import { useEffect, useState } from "react";
import { api, cls, fmt } from "../api.js";

// Auto mode: with an empty stock box the system trades today's top trending NIFTY 50 stocks - the ones whose MACD has
// been giving the most crosses and profit, and that are moving today. Made once a day after the close (15:30+) for the next session; shared by both systems.
// This panel shows that list and which list (yours or the automatic one) is in use right now.
export default function TrendingBox({ mode, onSelectSymbol }) {
  const [d, setD] = useState(null);

  useEffect(() => {
    let alive = true;
    const load = () => api(`/api/trending?mode=${mode}`).then((x) => alive && setD(x)).catch(() => {});
    load();
    const id = setInterval(load, 5000);
    return () => { alive = false; clearInterval(id); };
  }, [mode]);

  if (!d) return null;
  const when = d.picked_at ? d.picked_at.slice(11, 16) : null;
  const follows = !!d.follows;
  const nowList = d.stocks_now || [];

  if (d.whole_index) {
    return (
      <div className="card" style={{ marginBottom: 16 }}>
        <h3>Stocks traded: the whole {String(d.universe).toUpperCase()} ({nowList.length})</h3>
        <p className="hint" style={{ margin: 0 }}>
          {follows ? "This system trades the same stocks as the main system: " : ""}every stock of the index is monitored for MACD signals and traded from the
          first candle close after the open - there is no top-10 list. After the close (15:30+) each system loads and saves the candles of all of them for the next session.
        </p>
        {nowList.length > 0 && (
          <p className="hint" style={{ margin: "8px 0 0", lineHeight: 1.7 }}>
            {nowList.map((s) => (
              <span key={s} className={onSelectSymbol ? "clickable-row" : ""} style={{ marginRight: 10, cursor: onSelectSymbol ? "pointer" : "default" }}
                    onClick={() => onSelectSymbol?.(s)}>{s}</span>
            ))}
          </p>
        )}
      </div>
    );
  }

  return (
    <div className="card" style={{ marginBottom: 16 }}>
      <h3>{follows ? "Stocks traded today - the main system's list" : `Trending stocks today ${d.auto ? "(in use)" : "(not in use)"}`}</h3>
      <p className="hint" style={{ margin: "0 0 8px" }}>
        {follows
          ? `This system trades only the stocks the main system trades (${d.auto ? "its top trending stocks, made after the close for this session" : "its own stock list"}), ` +
            `but on 5 minute candles. ${nowList.length ? `Today: ${nowList.join(", ")}.` : "Nothing is picked yet - it starts trading once the main system has its list (before 09:15)."} ` +
            "The scores below are the main system's (computed on 15m history)."
          : d.auto
          ? `Your stock box is empty, so the system trades the top ${d.count} ${d.universe.toUpperCase()} stocks by MACD crosses and profit (${d.timeframe}), made once a day after the close (from their history) and shared with the 5 minute system, so both start at 09:15 with the same stocks.`
          : `Your own list of ${d.stock_list.length} stocks is in use. Clear the box above to let the system pick the top ${d.count} trending stocks instead.`}
        {" "}Only stocks on the active list are ever bought.
      </p>
      {d.picks.length === 0 ? (
        <p className="hint">No list yet - it is made after the close (15:30+) once the day's history has loaded, for the next session. If it is missing it is made before the open.</p>
      ) : (
        <>
          <p className="hint" style={{ margin: "0 0 6px" }}>Picked {d.date} at {when}.</p>
          <table>
            <thead><tr><th>#</th><th>Stock</th><th>Score</th><th>Price</th><th>Today</th><th>Crosses / day</th>
              <th>Trades</th><th>MACD return (10 d)</th><th>Whipsaw</th></tr></thead>
            <tbody>
              {d.picks.map((p) => (
                <tr key={p.symbol} className={onSelectSymbol ? "clickable-row" : ""} onClick={() => onSelectSymbol?.(p.symbol)}>
                  <td>{p.rank}</td><td>{p.symbol}</td><td>{fmt(p.score, 0)}</td><td>{fmt(p.price, 2)}</td>
                  <td className={cls(p.pct_change)}>{p.pct_change != null ? `${p.pct_change > 0 ? "+" : ""}${p.pct_change}%` : "-"}</td>
                  <td>{fmt(p.crosses_per_session, 1)}</td><td>{p.trades}</td>
                  <td className={cls(p.net_return_pct)}>{p.net_return_pct > 0 ? "+" : ""}{fmt(p.net_return_pct, 1)}%</td>
                  <td>{fmt(p.whipsaw_pct, 0)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}
