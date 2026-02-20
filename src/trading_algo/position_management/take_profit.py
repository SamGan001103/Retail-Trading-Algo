from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from trading_algo.core import BUY, SELL

TakeProfitOrderType = Literal["limit", "market"]


@dataclass(frozen=True)
class TakeProfitPlan:
    level: float
    ticks: int
    rrr: float
    order_type: TakeProfitOrderType
    rationale: str


@dataclass(frozen=True)
class TakeProfitPlanner:
    tick_size: float
    min_rrr: float = 3.0
    max_rrr: float = 10.0
    front_run_ticks: int = 2

    def plan(
        self,
        side: int,
        entry_price: float,
        risk_ticks: int,
        target_levels: list[float],
    ) -> TakeProfitPlan | None:
        if self.tick_size <= 0:
            return None
        if side not in (BUY, SELL):
            return None
        if risk_ticks <= 0:
            return None

        accepted: list[tuple[float, int, float]] = []
        for level in target_levels:
            if not isinstance(level, (int, float)):
                continue
            raw = float(level)
            if side == BUY:
                target = raw - self.front_run_ticks * self.tick_size
                if target <= entry_price:
                    continue
            else:
                target = raw + self.front_run_ticks * self.tick_size
                if target >= entry_price:
                    continue
            reward_ticks = int(math.ceil(abs(target - entry_price) / self.tick_size))
            if reward_ticks <= 0:
                continue
            rrr = reward_ticks / risk_ticks
            if self.min_rrr <= rrr <= self.max_rrr:
                accepted.append((target, reward_ticks, rrr))
        if not accepted:
            return None

        # Multiple nearby valid targets -> choose the median level.
        accepted.sort(key=lambda x: x[0])
        mid = len(accepted) // 2
        target, ticks, rrr = accepted[mid]
        return TakeProfitPlan(
            level=target,
            ticks=ticks,
            rrr=rrr,
            order_type="limit",
            rationale=f"median-valid-target(rrr={rrr:.2f})",
        )
