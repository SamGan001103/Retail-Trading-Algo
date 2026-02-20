from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from trading_algo.core import BUY, SELL

StopOrderType = Literal["stop_market"]


@dataclass(frozen=True)
class StopPlan:
    level: float
    ticks: int
    order_type: StopOrderType
    rationale: str


@dataclass(frozen=True)
class StopLossPlanner:
    tick_size: float
    noise_buffer_ticks: int = 2
    min_stop_ticks: int = 2
    max_stop_ticks: int = 200

    def plan(
        self,
        side: int,
        entry_price: float,
        invalidation_levels: list[float],
    ) -> StopPlan | None:
        if self.tick_size <= 0:
            return None
        if side not in (BUY, SELL):
            return None

        levels = [float(x) for x in invalidation_levels if isinstance(x, (int, float))]
        if side == BUY:
            levels = [x for x in levels if x < entry_price]
        else:
            levels = [x for x in levels if x > entry_price]
        if not levels:
            return None

        # Closest invalidation is usually the cleanest "idea is wrong" boundary.
        raw = max(levels) if side == BUY else min(levels)
        if side == BUY:
            stop = raw - self.noise_buffer_ticks * self.tick_size
        else:
            stop = raw + self.noise_buffer_ticks * self.tick_size
        if side == BUY and stop >= entry_price:
            return None
        if side == SELL and stop <= entry_price:
            return None

        ticks = int(math.ceil(abs(entry_price - stop) / self.tick_size))
        ticks = max(self.min_stop_ticks, ticks)
        if ticks > self.max_stop_ticks:
            return None

        return StopPlan(
            level=stop,
            ticks=ticks,
            order_type="stop_market",
            rationale=f"invalidation@{raw:.5f}+noise({self.noise_buffer_ticks}t)",
        )
