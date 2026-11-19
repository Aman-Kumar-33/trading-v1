"""A daily kill switch: once tripped, blocks every new entry for the rest of the trading day. It never
auto-clears mid-day - only an explicit `reset()` (called once, at the start of the next trading day)
turns it back off. That's deliberate: a daily loss limit or a consecutive-loss streak is exactly the
situation where the system should stop digging, not quietly resume once the numbers look better."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KillSwitch:
    tripped: bool = False
    reason: str = ""

    def trip(self, reason: str) -> None:
        if not self.tripped:   # keep the FIRST reason, not whatever trips it again afterward
            self.tripped = True
            self.reason = reason

    def reset(self) -> None:
        self.tripped = False
        self.reason = ""
