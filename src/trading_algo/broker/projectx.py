from __future__ import annotations

from typing import Any

from trading_algo.api import ProjectXClient, resolve_contract_id, search_contracts
from trading_algo.execution import ExecutionEngine

from .projectx_realtime import ProjectXRealtimeStream


class ProjectXBrokerAdapter:
    def __init__(self, base_url: str, username: str, api_key: str, user_hub_url: str, market_hub_url: str):
        self._client = ProjectXClient(base_url=base_url, username=username, api_key=api_key)
        self._user_hub_url = user_hub_url
        self._market_hub_url = market_hub_url
        self._engines: dict[int, ExecutionEngine] = {}

    def _engine(self, account_id: int) -> ExecutionEngine:
        key = int(account_id)
        engine = self._engines.get(key)
        if engine is None:
            engine = ExecutionEngine(self._client, account_id=key)
            self._engines[key] = engine
        return engine

    def resolve_contract_id(self, symbol: str, live: bool) -> str:
        return resolve_contract_id(self._client, symbol, live)

    def search_contracts(self, search_text: str, live: bool) -> list[dict[str, Any]]:
        return search_contracts(self._client, search_text=search_text, live=live)

    def list_accounts(self) -> list[dict[str, Any]]:
        data = self._client.post_json("/api/Account/search", {"onlyActiveAccounts": True}, "ACCOUNT_SEARCH")
        if not data.get("success"):
            raise RuntimeError(f"Account search failed: {data}")
        return data.get("accounts") or []

    def create_stream(self, account_id: int) -> ProjectXRealtimeStream:
        return ProjectXRealtimeStream(
            client=self._client,
            account_id=account_id,
            user_hub_url=self._user_hub_url,
            market_hub_url=self._market_hub_url,
        )

    def open_orders(self, account_id: int) -> list[dict[str, Any]]:
        return self._engine(account_id).open_orders()

    def open_positions(self, account_id: int) -> list[dict[str, Any]]:
        return self._engine(account_id).open_positions()

    def cancel_order(self, account_id: int, order_id: int) -> dict[str, Any]:
        return self._engine(account_id).cancel_order(order_id)

    def close_contract(self, account_id: int, contract_id: str) -> dict[str, Any]:
        return self._engine(account_id).close_contract(contract_id)

    def flatten(self, account_id: int) -> tuple[int, int]:
        return self._engine(account_id).flatten()

    def place_market_with_brackets(
        self,
        account_id: int,
        contract_id: str,
        side: int,
        size: int,
        sl_ticks_abs: int,
        tp_ticks_abs: int,
    ) -> dict[str, Any]:
        return self._engine(account_id).place_market_with_brackets(
            contract_id=contract_id,
            side=side,
            size=size,
            sl_ticks_abs=sl_ticks_abs,
            tp_ticks_abs=tp_ticks_abs,
        )

    def close(self) -> None:
        self._client.close()
