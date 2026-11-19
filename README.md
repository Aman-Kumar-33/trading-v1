# trading-v1 — MACD paper-trading systems (NSE / Angel One)

Two sandbox paper-trading systems that trade the whole NIFTY 50 with MACD(12,26,9) only:

| System | Config | Port | Candle | Signal mode |
|---|---|---|---|---|
| Main (15m) | `config/settings.yaml` | 8766 | 15m | `bar_close` |
| 5 minute | `config/settings_5m.yaml` | 8767 | 5m | `candle_confirmed` (tolerance max(0.02, 0.02×ATR), 5% stop) |

* Trading 09:15–15:00 (square-off 15:00, intraday only), at most 15 open positions per system.
* After 15:30 each system saves the 5m and 15m history of all 50 stocks for the next session.
* `scripts/both_systems.py` starts both and opens two browser tabs (built as `both_systems.exe` with PyInstaller, see `build_one/both_systems.spec`).

Setup: copy `.env.example` to `.env` and fill in the Angel One / Telegram credentials; `pip install -r requirements.txt`;
`cd frontend/dashboard-react && npm install && npm run build`; run `python scripts/both_systems.py`.
Tests: `python -m pytest tests` (some need recorded 1-minute data that is not in the repo).
