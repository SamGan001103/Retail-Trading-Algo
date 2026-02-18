from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskLimits:
    max_open_positions: int = 1
    max_open_orders_while_flat: int = 0


def enforce_position_limits(open_positions: int, open_orders: int, limits: RiskLimits) -> tuple[bool, str]:
    if open_positions > limits.max_open_positions:
        return False, f"Too many positions: {open_positions}>{limits.max_open_positions}"
    if open_positions == 0 and open_orders > limits.max_open_orders_while_flat:
        return False, f"Stale open orders while flat: {open_orders}>{limits.max_open_orders_while_flat}"
    return True, "OK"

