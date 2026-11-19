import { useEffect, useRef } from "react";
import {
  createChart, CandlestickSeries, LineSeries, HistogramSeries, createSeriesMarkers, ColorType,
} from "lightweight-charts";
import { fmt } from "../api.js";

const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// lightweight-charts draws every intraday label in UTC. Timestamps from the backend are IST
// ("YYYY-MM-DD HH:MM:SS+05:30"), so shift by +5:30 - a 09:15 bar then reads 09:15, exactly like a broker chart.
const IST_OFFSET_SECONDS = 19800;
function toChartTime(ts) {
  return Math.floor(new Date(ts.replace(" ", "T")).getTime() / 1000) + IST_OFFSET_SECONDS;
}

const VISIBLE_BARS = { "1m": 150, "5m": 150, "15m": 130, "30m": 130, "1h": 120, "1d": 120 };

// Price on top, MACD(12,26,9) underneath. BUY/SELL arrows sit on the bar where the MACD line crosses its
// signal line (the bar whose CLOSE completes the cross) - the same bars the strategy fires on. IN/OUT dots
// are the trades the system actually took.
export default function StockChart({ candles, macd = [], signals = [], trades = [], timeframe = "15m", height = 560 }) {
  const containerRef = useRef(null);
  const tooltipRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const green = cssVar("--green") || "#2ecc71";
    const red = cssVar("--red") || "#ff5c5c";
    const chart = createChart(containerRef.current, {
      height,
      layout: { background: { type: ColorType.Solid, color: cssVar("--panel") || "#0d0f12" },
               textColor: cssVar("--muted") || "#7d8694", panes: { separatorColor: cssVar("--border") || "#22262d" } },
      grid: { vertLines: { color: cssVar("--border") || "#1c1f24" }, horzLines: { color: cssVar("--border") || "#1c1f24" } },
      timeScale: { timeVisible: true, secondsVisible: false, rightOffset: 4 },
      crosshair: { mode: 0 },
    });
    const price = chart.addSeries(CandlestickSeries, {
      upColor: green, downColor: red, borderVisible: false, wickUpColor: green, wickDownColor: red,
    }, 0);
    const hist = chart.addSeries(HistogramSeries, { priceLineVisible: false, lastValueVisible: false }, 1);
    const macdLine = chart.addSeries(LineSeries, { color: "#4f8cff", lineWidth: 2, priceLineVisible: false, title: "MACD" }, 1);
    const signalLine = chart.addSeries(LineSeries, { color: "#ffb020", lineWidth: 2, priceLineVisible: false, title: "signal" }, 1);
    try { chart.panes()[1].setStretchFactor(0.45); } catch { /* older API - default split is fine */ }
    chartRef.current = { chart, price, hist, macdLine, signalLine };

    const onResize = () => chart.applyOptions({ width: containerRef.current.clientWidth });
    onResize();
    window.addEventListener("resize", onResize);
    return () => { window.removeEventListener("resize", onResize); chart.remove(); chartRef.current = null; };
  }, [height]);

  useEffect(() => {
    const ref = chartRef.current;
    if (!ref || !candles || !candles.length) return;
    const green = cssVar("--green") || "#2ecc71";
    const red = cssVar("--red") || "#ff5c5c";
    const bars = candles.map((c) => ({ time: toChartTime(c.ts), open: c.open, high: c.high, low: c.low, close: c.close }));
    ref.price.setData(bars);
    ref.hist.setData(macd.map((m) => ({ time: toChartTime(m.ts), value: m.hist,
                                        color: m.hist >= 0 ? `${green}99` : `${red}99` })));
    ref.macdLine.setData(macd.map((m) => ({ time: toChartTime(m.ts), value: m.macd })));
    ref.signalLine.setData(macd.map((m) => ({ time: toChartTime(m.ts), value: m.signal })));

    // a trade at exactly a bar boundary was executed at that bar's CLOSE (= the next bar's start), so it
    // belongs to the bar that just ended; a tick-time exit inside a bar belongs to the bar containing it.
    // Both are "the last bar that starts strictly before the timestamp".
    const barTimeFor = (ts) => {
      const target = toChartTime(ts);
      let best = bars[0].time;
      for (const b of bars) { if (b.time < target) best = b.time; else break; }
      return best;
    };

    const details = {};   // "<kind>-<time>" -> popup text
    const markers = [];
    for (const s of signals) {
      const time = toChartTime(s.ts);
      const buy = s.side === "BUY";
      markers.push({ time, position: buy ? "belowBar" : "aboveBar", color: buy ? green : red,
                    shape: buy ? "arrowUp" : "arrowDown", text: buy ? "BUY" : "SELL" });
      details[`sig-${time}`] = `MACD ${buy ? "bullish" : "bearish"} cross on the candle starting ${s.ts.slice(11, 16)} - the order goes out when it closes, ` +
                               `at ${fmt(s.close, 2)}; MACD ${s.macd} ${buy ? ">" : "<"} signal ${s.signal}`;
    }
    for (const t of trades) {
      if (t.entry_ts) {
        const time = barTimeFor(t.entry_ts);
        markers.push({ time, position: "inBar", color: "#4f8cff", shape: "circle", text: "IN" });
        details[`in-${time}`] = `${t.strategy_id} - ${t.side} filled at ${t.entry_ts.slice(11, 19)} @ ${fmt(t.entry_price, 2)} x ${t.qty} (the close of this candle)`;
      }
      if (t.exit_ts) {
        const time = barTimeFor(t.exit_ts);
        markers.push({ time, position: "inBar", color: (t.pnl_net ?? 0) >= 0 ? green : red, shape: "square", text: "OUT" });
        details[`out-${time}`] = `${t.strategy_id} - exit (${t.exit_reason}) at ${t.exit_ts.slice(11, 19)} @ ${fmt(t.exit_price, 2)} - net ${fmt(t.pnl_net, 0)}`;
      }
    }
    markers.sort((a, b) => a.time - b.time);
    const markersApi = createSeriesMarkers(ref.price, markers);

    const handleClick = (param) => {
      const tip = tooltipRef.current;
      if (!tip) return;
      const hit = param.time && (details[`out-${param.time}`] || details[`in-${param.time}`] || details[`sig-${param.time}`]);
      if (!hit) { tip.style.display = "none"; return; }
      tip.textContent = hit;
      tip.style.display = "block";
      tip.style.left = `${Math.max(0, param.point.x - 120)}px`;
      tip.style.top = `${param.point.y + 12}px`;
    };
    ref.chart.subscribeClick(handleClick);

    const n = bars.length;
    const visible = Math.min(n, VISIBLE_BARS[timeframe] ?? 130);
    ref.chart.timeScale().setVisibleLogicalRange({ from: n - visible, to: n + 3 });
    return () => { ref.chart.unsubscribeClick(handleClick); markersApi.detach(); };
  }, [candles, macd, signals, trades, timeframe]);

  return (
    <div className="chart-area" style={{ position: "relative" }}>
      <div ref={containerRef} />
      <div ref={tooltipRef} className="chart-tooltip" style={{ display: "none" }} />
      <div className="chart-legend">
        <span className="legend-item"><span className="dot" style={{ background: "var(--green)" }} />BUY = MACD crosses above signal</span>
        <span className="legend-item"><span className="dot" style={{ background: "var(--red)" }} />SELL = MACD crosses below signal</span>
        <span className="legend-item"><span className="dot" style={{ background: "#4f8cff" }} />IN / OUT = fills taken</span>
        <span className="hint">times are IST; a candle is labelled with its start, and an order is placed when it closes - so a cross on the 12:50 candle (5m) is bought at 12:55:00, and the trade tables show 12:55 - click a marker for details - drag to pan, scroll to zoom</span>
      </div>
    </div>
  );
}
