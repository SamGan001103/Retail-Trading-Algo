from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from trading_algo.core import BUY, SELL


@dataclass(frozen=True)
class Brackets:
    sl_ticks: int
    tp_ticks: int


class SupportsPostJson(Protocol):
    def post_json(self, path: str, payload: dict[str, Any], label: str, timeout: int = 30) -> dict[str, Any]:
        ...


def sign_brackets(side: int, sl_abs: int, tp_abs: int) -> Brackets:
    sl = abs(int(sl_abs))
    tp = abs(int(tp_abs))
    if side == BUY:
        return Brackets(sl_ticks=-sl, tp_ticks=+tp)
    if side == SELL:
        return Brackets(sl_ticks=+sl, tp_ticks=-tp)
    raise ValueError(f"Invalid side={side} (expected 0=BUY or 1=SELL)")


class ExecutionEngine:
    def __init__(self, client: SupportsPostJson, account_id: int):
        self.client = client
        self.account_id = int(account_id)

    def open_orders(self) -> list[dict[str, Any]]:
        data = self.client.post_json("/api/Order/searchOpen", {"accountId": self.account_id}, "ORDER_SEARCH_OPEN")
        if not data.get("success"):
            raise RuntimeError(data)
        return data.get("orders") or []

    def open_positions(self) -> list[dict[str, Any]]:
        data = self.client.post_json(
            "/api/Position/searchOpen", {"accountId": self.account_id}, "POSITION_SEARCH_OPEN"
        )
        if not data.get("success"):
            raise RuntimeError(data)
        return data.get("positions") or []

    def snapshot(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        return self.open_orders(), self.open_positions()

    def cancel_order(self, order_id: int) -> dict[str, Any]:
        return self.client.post_json(
            "/api/Order/cancel",
            {"accountId": self.account_id, "orderId": int(order_id)},
            f"ORDER_CANCEL({order_id})",
        )

    def close_contract(self, contract_id: str) -> dict[str, Any]:
        return self.client.post_json(
            "/api/Position/closeContract",
            {"accountId": self.account_id, "contractId": str(contract_id)},
            f"POSITION_CLOSE_CONTRACT({contract_id})",
        )

    def flatten(self) -> tuple[int, int]:
        orders = self.open_orders()
        cancel_attempts = 0
        for order in orders:
            order_id = order.get("id")
            if order_id is None:
                continue
            cancel_attempts += 1
            response = self.cancel_order(int(order_id))
            if not response.get("success"):
                print("cancel failed:", response)

        positions = self.open_positions()
        close_attempts = 0
        for position in positions:
            contract_id = position.get("contractId")
            if not contract_id:
                continue
            close_attempts += 1
            response = self.close_contract(str(contract_id))
            if not response.get("success"):
                print("close failed:", response)

        return cancel_attempts, close_attempts

    def can_enter_trade(self, contract_id: str | None = None) -> tuple[bool, str]:
        positions = self.open_positions()
        orders = self.open_orders()
        if contract_id:
            positions = [p for p in positions if p.get("contractId") == contract_id]
            orders = [o for o in orders if o.get("contractId") == contract_id]
        if positions:
            return False, f"Blocked: in position(s)={len(positions)}"
        if orders:
            return False, f"Blocked: {len(orders)} open order(s) while flat"
        return True, "OK"

    def place_market_with_brackets(
        self,
        contract_id: str,
        side: int,
        size: int,
        sl_ticks_abs: int,
        tp_ticks_abs: int,
    ) -> dict[str, Any]:
        brackets = sign_brackets(side, sl_ticks_abs, tp_ticks_abs)
        payload = {
            "accountId": self.account_id,
            "contractId": str(contract_id),
            "type": 2,
            "side": int(side),
            "size": int(size),
            "stopLossBracket": {"ticks": int(brackets.sl_ticks), "type": 4},
            "takeProfitBracket": {"ticks": int(brackets.tp_ticks), "type": 1},
        }
        return self.client.post_json("/api/Order/place", payload, "ORDER_PLACE")

    def has_position(self, contract_id: str) -> bool:
        return any(position.get("contractId") == contract_id for position in self.open_positions())

    def open_orders_for_contract(self, contract_id: str) -> list[dict[str, Any]]:
        return [order for order in self.open_orders() if order.get("contractId") == contract_id]

    def verify_brackets_present(self, contract_id: str, expected: int = 2) -> tuple[bool, str]:
        if not self.has_position(contract_id):
            return True, "Flat (no brackets required)"
        orders = self.open_orders_for_contract(contract_id)
        if len(orders) >= expected:
            return True, f"OK: {len(orders)} bracket order(s) working"
        return False, f"Missing brackets: only {len(orders)} open order(s)"

