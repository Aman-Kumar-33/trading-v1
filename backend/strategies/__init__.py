"""Second (sandbox) system's strategy pool: MACD only for now, no allocator/risk-manager gating. The MFI
strategies are kept on disk but commented out of the pool - uncomment the two lines (and the imports) to
bring them back. See backend/execution/strategy_runner.py for the simplified live wiring."""
from __future__ import annotations

from backend.strategies.base import Strategy
# from backend.strategies.intraday.macd_mfi_hybrid import MacdMfiHybrid   # MFI paused - MACD only for now
from backend.strategies.intraday.macd_only import MacdOnly
# from backend.strategies.intraday.mfi_only import MfiOnly                # MFI paused - MACD only for now

INTRADAY_STRATEGIES: list[type[Strategy]] = [MacdOnly]   # was [MacdOnly, MfiOnly, MacdMfiHybrid]
LONGTERM_STRATEGIES: list[type[Strategy]] = []
ALL_STRATEGIES: list[type[Strategy]] = INTRADAY_STRATEGIES + LONGTERM_STRATEGIES
STRATEGY_BY_ID: dict[str, type[Strategy]] = {cls.strategy_id: cls for cls in ALL_STRATEGIES}

__all__ = ["Strategy", "INTRADAY_STRATEGIES", "LONGTERM_STRATEGIES", "ALL_STRATEGIES", "STRATEGY_BY_ID"]
