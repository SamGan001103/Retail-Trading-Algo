from __future__ import annotations

from trading_algo.api.client import ProjectXClient


def search_contracts(client: ProjectXClient, search_text: str, live: bool) -> list[dict]:
    data = client.post_json("/api/Contract/search", {"live": bool(live), "searchText": search_text}, "CONTRACT_SEARCH")
    if not data.get("success"):
        raise RuntimeError(f"Contract search failed: {data}")
    return data.get("contracts") or []


def resolve_contract_id(client: ProjectXClient, search_text: str, live: bool) -> str:
    contracts = search_contracts(client, search_text, live)
    if not contracts:
        raise RuntimeError(f"No contracts returned for searchText={search_text!r} live={live}")
    return contracts[0]["id"]

