"""Persistence for live/paper trading: every closed trade goes to the SQLite `trades` table (what the weekly
review and the dashboard read), and the current open positions are written to a JSON state file after
every change. The state file is what lets a long carried overnight survive the process being stopped
after the close and restarted next morning.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from backend.core.models import Horizon, Mode, Side

log = logging.getLogger(__name__)


class TradeJournal:
    def __init__(self, conn: sqlite3.Connection | None, state_file: Path, mode: Mode):
        self.conn = conn
        self.state_file = Path(state_file)
        self.mode = mode

    def record_trade(self, trade) -> None:
        if self.conn is None:
            return
        p = trade.position
        self.conn.execute(
            """INSERT INTO trades (mode, strategy_id, strategy_version, symbol, side, horizon, qty, entry_ts,
                                   entry_price, exit_ts, exit_price, stop_loss, target, pnl_gross, costs,
                                   pnl_net, r_multiple, exit_reason)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (self.mode.value, p.strategy_id, p.strategy_version, p.symbol, p.side.value, p.horizon.value, p.qty,
             p.opened_at.isoformat(), p.entry_price, trade.closed_at.isoformat(), trade.exit_price, p.stop_loss,
             p.target, trade.pnl_gross, trade.costs, trade.pnl_net, trade.r_multiple, trade.exit_reason))

    def save_positions(self, positions: list) -> None:
        rows = []
        for p in positions:
            d = asdict(p)
            d["side"], d["horizon"], d["opened_at"] = p.side.value, p.horizon.value, p.opened_at.isoformat()
            rows.append(d)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        tmp.replace(self.state_file)

    def load_positions(self) -> list[dict]:
        if not self.state_file.exists():
            return []
        rows = json.loads(self.state_file.read_text(encoding="utf-8"))
        for d in rows:
            d["side"], d["horizon"] = Side(d["side"]), Horizon(d["horizon"])
            d["opened_at"] = datetime.fromisoformat(d["opened_at"])
        return rows
