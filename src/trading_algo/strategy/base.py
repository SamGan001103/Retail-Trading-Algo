from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyDecision:
    should_enter: bool
    side: int
    size: int
    reason: str

