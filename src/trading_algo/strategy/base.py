from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class StrategyDecision:
    should_exit: bool
    should_enter: bool
    side: int
    size: int
    reason: str
    sl_ticks_abs: int | None = None
    tp_ticks_abs: int | None = None


@dataclass(frozen=True)
class MarketBar:
    ts: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class StrategyContext:
    index: int
    total_bars: int


@dataclass(frozen=True)
class PositionState:
    in_position: bool
    side: int | None = None
    size: int = 0
    entry_price: float | None = None
    bars_in_position: int = 0


class Strategy(Protocol):
    def on_bar(self, bar: MarketBar, context: StrategyContext, position: PositionState) -> StrategyDecision:
        ...

