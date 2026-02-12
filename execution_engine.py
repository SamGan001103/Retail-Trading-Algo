# execution_engine.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from projectx_api import login_key, post_json

# ProjectX side enum
BUY = 0
SELL = 1


@dataclass(frozen=True)
class Brackets:
    """
    Signed tick offsets relative to entry price.
    LONG:  SL negative, TP positive
    SHORT: SL positive, TP negative
    """
    sl_ticks: int
    tp_ticks: int


def sign_brackets(side: int, sl_abs: int, tp_abs: int) -> Brackets:
    sl = abs(int(sl_abs))
    tp = abs(int(tp_abs))

    if side == BUY:      # long
        return Brackets(sl_ticks=-sl, tp_ticks=+tp)
    if side == SELL:     # short
        return Brackets(sl_ticks=+sl, tp_ticks=-tp)

    raise ValueError(f"Invalid side={side} (expected 0=BUY or 1=SELL)")


class ExecutionEngine:
    """
    Deterministic wrapper around ProjectX execution-control endpoints.
    Focus: correctness + safety. Designed to be used by bot runtime code.

    Notes on latency:
    - These are REST calls; do NOT put them in a high-frequency hot path.
    - Cache auth token to avoid re-auth on every call.
    """

    def __init__(self, base_url: str, username: str, api_key: str, account_id: int, token_ttl_sec: int = 25 * 60):
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.api_key = api_key
        self.account_id = int(account_id)

        # Token cache (conservative TTL; can be improved later by decoding JWT exp)
        self._token_cache: str | None = None
        self._token_expiry_ts: float = 0.0
        self._token_ttl_sec = int(token_ttl_sec)

    # ---------- Auth ----------
    def _token(self) -> str:
        now = time.time()
        if self._token_cache and now < self._token_expiry_ts:
            return self._token_cache

        tok = login_key(self.base_url, self.username, self.api_key)
        self._token_cache = tok
        self._token_expiry_ts = now + self._token_ttl_sec
        return tok

    def _post(self, path: str, payload: Dict[str, Any], label: str) -> Dict[str, Any]:
        token = self._token()
        data = post_json(self.base_url, token, path, payload, label)
        # Optional: if we ever get unauthorized, we can refresh and retry once.
        # We'll keep it simple for now.
        return data

    # ---------- Queries ----------
    def open_orders(self) -> List[Dict[str, Any]]:
        data = self._post("/api/Order/searchOpen", {"accountId": self.account_id}, "ORDER_SEARCH_OPEN")
        if not data.get("success"):
            raise RuntimeError(data)
        return data.get("orders") or []

    def open_positions(self) -> List[Dict[str, Any]]:
        data = self._post("/api/Position/searchOpen", {"accountId": self.account_id}, "POSITION_SEARCH_OPEN")
        if not data.get("success"):
            raise RuntimeError(data)
        return data.get("positions") or []
    
    def snapshot(self) -> tuple[list[dict], list[dict]]:
        """Returns (open_orders, open_positions)."""
        return self.open_orders(), self.open_positions()


    # ---------- Controls ----------
    def cancel_order(self, order_id: int) -> Dict[str, Any]:
        return self._post(
            "/api/Order/cancel",
            {"accountId": self.account_id, "orderId": int(order_id)},
            f"ORDER_CANCEL({order_id})",
        )

    def close_contract(self, contract_id: str) -> Dict[str, Any]:
        return self._post(
            "/api/Position/closeContract",
            {"accountId": self.account_id, "contractId": str(contract_id)},
            f"POSITION_CLOSE_CONTRACT({contract_id})",
        )

    def flatten(self) -> Tuple[int, int]:
        """
        Kill switch:
          1) Cancel all open orders
          2) Close all open positions (per contract)
        Returns: (num_order_cancel_attempts, num_position_close_attempts)
        """
        orders = self.open_orders()
        cancel_n = 0
        for o in orders:
            oid = o.get("id")
            if oid is None:
                continue
            cancel_n += 1
            resp = self.cancel_order(int(oid))
            if not resp.get("success"):
                # Keep going; best-effort flatten
                print("⚠️ cancel failed:", resp)

        positions = self.open_positions()
        close_n = 0
        for p in positions:
            cid = p.get("contractId")
            if not cid:
                continue
            close_n += 1
            resp = self.close_contract(str(cid))
            if not resp.get("success"):
                print("⚠️ close failed:", resp)

        return cancel_n, close_n

    def can_enter_trade(self, contract_id: str | None = None) -> tuple[bool, str]:
        """
        Position-aware gate.

        - If any open position exists: block new entry (one-position rule).
        - If flat: block if any open orders exist (stale orders protection).
        - Optional: if contract_id provided, only consider orders/positions for that contract.
        """
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


    # ---------- Placement ----------
    def place_market_with_brackets(
        self,
        contract_id: str,
        side: int,
        size: int,
        sl_ticks_abs: int,
        tp_ticks_abs: int,
    ) -> Dict[str, Any]:
        """
        Places a market order with attached stop-loss and take-profit brackets.

        Requires your TopstepX account to have Auto-OCO Brackets enabled.
        Brackets use signed tick offsets (ProjectX enforces sign).
        """
        b = sign_brackets(side, sl_ticks_abs, tp_ticks_abs)

        payload = {
            "accountId": self.account_id,
            "contractId": str(contract_id),
            "type": 2,          # Market
            "side": int(side),  # 0=BUY, 1=SELL
            "size": int(size),
            "stopLossBracket": {"ticks": int(b.sl_ticks), "type": 4},   # Stop
            "takeProfitBracket": {"ticks": int(b.tp_ticks), "type": 1}, # Limit
        }

        data = self._post("/api/Order/place", payload, "ORDER_PLACE")
        return data
    


    def has_position(self, contract_id: str) -> bool:
        pos = self.open_positions()
        return any(p.get("contractId") == contract_id for p in pos)



    def open_orders_for_contract(self, contract_id: str) -> list[dict]:
        return [o for o in self.open_orders() if o.get("contractId") == contract_id]



    def verify_brackets_present(self, contract_id: str, expected: int = 2) -> tuple[bool, str]:
        """
        If in a position, ensure exit orders exist.
        With Auto-OCO we usually expect 2 working orders (SL + TP).
        """
        if not self.has_position(contract_id):
            return True, "Flat (no brackets required)"

        orders = self.open_orders_for_contract(contract_id)
        if len(orders) >= expected:
            return True, f"OK: {len(orders)} bracket order(s) working"
        return False, f"Missing brackets: only {len(orders)} open order(s) for contract while in position"

