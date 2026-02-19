from __future__ import annotations

from typing import Any, Protocol


class BrokerStream(Protocol):
    # Stream snapshots should use canonical keys where possible:
    # - orders: id, contractId
    # - positions: id, contractId
    def start(self, contract_id: str) -> None:
        ...

    def stop(self) -> None:
        ...

    def snapshot(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        ...

    def last_quote(self, contract_id: str) -> dict[str, Any] | None:
        ...

    def last_trade(self, contract_id: str) -> dict[str, Any] | None:
        ...

    def last_depth(self, contract_id: str) -> dict[str, Any] | None:
        ...


class BrokerAdapter(Protocol):
    def resolve_contract_id(self, symbol: str, live: bool) -> str:
        ...

    def search_contracts(self, search_text: str, live: bool) -> list[dict[str, Any]]:
        ...

    def list_accounts(self) -> list[dict[str, Any]]:
        ...

    def create_stream(self, account_id: int) -> BrokerStream:
        ...

    def open_orders(self, account_id: int) -> list[dict[str, Any]]:
        ...

    def open_positions(self, account_id: int) -> list[dict[str, Any]]:
        ...

    def cancel_order(self, account_id: int, order_id: int) -> dict[str, Any]:
        ...

    def close_contract(self, account_id: int, contract_id: str) -> dict[str, Any]:
        ...

    def flatten(self, account_id: int) -> tuple[int, int]:
        ...

    def place_market_with_brackets(
        self,
        account_id: int,
        contract_id: str,
        side: int,
        size: int,
        sl_ticks_abs: int,
        tp_ticks_abs: int,
    ) -> dict[str, Any]:
        ...

    def close(self) -> None:
        ...
