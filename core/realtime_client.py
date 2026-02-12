# realtime_client.py (signalrcore + Record Separator fix)
from __future__ import annotations

import os
import threading
import time
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from signalrcore.hub_connection_builder import HubConnectionBuilder
from signalrcore.protocol.json_hub_protocol import JsonHubProtocol

from projectx_api import login_key  # type: ignore


def env_bool(name: str, default: bool = False) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "y", "on")


DEBUG_RTC = env_bool("DEBUG_RTC", False)
SUB_DEPTH = env_bool("SUB_DEPTH", False)  # optional; keep False for lower bandwidth


@dataclass
class RTState:
    orders: Dict[int, Dict[str, Any]]
    positions: Dict[int, Dict[str, Any]]
    last_quote_by_contract: Dict[str, Dict[str, Any]]
    last_trade_by_contract: Dict[str, Dict[str, Any]]


class JsonHubProtocolRS(JsonHubProtocol):
    """
    Robust SignalR JSON protocol with Record Separator framing + buffering.

    - SignalR text frames end each message with RS = 0x1E ('\\x1e')
    - WebSocket transport may deliver partial chunks
    - We buffer until we see RS, then parse complete messages
    - We delegate parsing to signalrcore so we return HubMessage objects (with .type)
    """
    RS = "\x1e"

    def __init__(self):
        super().__init__()
        self._buf = ""

    def parse_messages(self, raw: str):
        if not raw:
            return []

        # Append to buffer (handles fragmented websocket payloads)
        self._buf += raw

        # Split into complete frames; last part may be incomplete remainder
        parts = self._buf.split(self.RS)
        self._buf = parts.pop()  # remainder without RS (possibly "")

        out = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            # Let signalrcore create proper message objects
            out.extend(super().parse_messages(part))

        # Safety: avoid unbounded growth if server misbehaves
        if len(self._buf) > 2_000_000:
            # drop buffer to prevent runaway memory
            self._buf = ""

        return out


class RealtimeClient:
    """
    Realtime client using signalrcore (recommended).

    Connect to hub endpoints:
      - https://.../hubs/user
      - https://.../hubs/market

    SignalR negotiates WebSocket transport internally (wss).
    Token is passed via query param (?access_token=...).
    """

    def __init__(
        self,
        base_url: str,
        username: str,
        api_key: str,
        account_id: int,
        user_hub_url: str,
        market_hub_url: str,
    ):
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.api_key = api_key
        self.account_id = int(account_id)

        self.user_hub_url = user_hub_url.rstrip("/")
        self.market_hub_url = market_hub_url.rstrip("/")

        self._lock = threading.RLock()
        self._state = RTState(orders={}, positions={}, last_quote_by_contract={}, last_trade_by_contract={})

        self._running = False
        self._stop_evt = threading.Event()

        self._contract_id: Optional[str] = None

        self._user_conn = None
        self._mkt_conn = None

        self._user_connected = threading.Event()
        self._mkt_connected = threading.Event()

    def _log(self, *a: Any) -> None:
        if DEBUG_RTC:
            print("[RTC]", *a)

    def _token(self) -> str:
        return login_key(self.base_url, self.username, self.api_key)

    # ---- handlers (payload normalization) ----
    @staticmethod
    def _iter_dicts(x: Any):
        if isinstance(x, dict):
            yield x
        elif isinstance(x, list):
            for item in x:
                if isinstance(item, dict):
                    yield item

    def _on_user_order(self, args: List[Any]) -> None:
        payload = args[0] if len(args) == 1 else args
        for d in self._iter_dicts(payload):
            oid = d.get("id")
            if oid is None:
                continue
            with self._lock:
                self._state.orders[int(oid)] = d

    def _on_user_position(self, args: List[Any]) -> None:
        payload = args[0] if len(args) == 1 else args
        for d in self._iter_dicts(payload):
            pid = d.get("id")
            if pid is None:
                continue
            with self._lock:
                self._state.positions[int(pid)] = d

    def _on_quote(self, args: List[Any]) -> None:
        # expected: [contractId, quoteObj]
        if len(args) < 2:
            return
        contract_id, data = args[0], args[1]
        if not isinstance(data, dict):
            return
        with self._lock:
            self._state.last_quote_by_contract[str(contract_id)] = data

    def _on_trade(self, args: List[Any]) -> None:
        # expected: [contractId, tradeObj]
        if len(args) < 2:
            return
        contract_id, data = args[0], args[1]
        if not isinstance(data, dict):
            return
        with self._lock:
            self._state.last_trade_by_contract[str(contract_id)] = data

    # ---- public API ----
    def start(self, contract_id: str) -> None:
        if self._running:
            return
        self._running = True
        self._contract_id = str(contract_id)

        self._stop_evt.clear()
        self._user_connected.clear()
        self._mkt_connected.clear()

        token = self._token()
        user_url = f"{self.user_hub_url}?access_token={token}"
        mkt_url = f"{self.market_hub_url}?access_token={token}"

        self._log("Connecting user hub:", self.user_hub_url)
        self._log("Connecting market hub:", self.market_hub_url)

        # Build hub connections (use RS-aware protocol to avoid JSONDecodeError)
        self._user_conn = (
            HubConnectionBuilder()
            .with_url(user_url)
            .with_hub_protocol(JsonHubProtocolRS())
            .with_automatic_reconnect(
                {
                    "type": "raw",
                    "keep_alive_interval": 10,
                    "reconnect_interval": 5,
                    "max_attempts": 999999,
                }
            )
            .build()
        )

        self._mkt_conn = (
            HubConnectionBuilder()
            .with_url(mkt_url)
            .with_hub_protocol(JsonHubProtocolRS())
            .with_automatic_reconnect(
                {
                    "type": "raw",
                    "keep_alive_interval": 10,
                    "reconnect_interval": 5,
                    "max_attempts": 999999,
                }
            )
            .build()
        )

        # Register callbacks
        self._user_conn.on_open(lambda: (self._log("User hub connected"), self._user_connected.set(), self._send_subscriptions()))
        self._mkt_conn.on_open(lambda: (self._log("Market hub connected"), self._mkt_connected.set(), self._send_subscriptions()))

        self._user_conn.on_close(lambda: self._log("User hub closed"))
        self._mkt_conn.on_close(lambda: self._log("Market hub closed"))

        self._user_conn.on_error(lambda data: self._log("User hub error:", data))
        self._mkt_conn.on_error(lambda data: self._log("Market hub error:", data))

        # Server event handlers (method names must match ProjectX)
        self._user_conn.on("GatewayUserOrder", self._on_user_order)
        self._user_conn.on("GatewayUserPosition", self._on_user_position)

        self._mkt_conn.on("GatewayQuote", self._on_quote)
        self._mkt_conn.on("GatewayTrade", self._on_trade)

        # Start connections (signalrcore manages its own threads)
        self._user_conn.start()
        self._mkt_conn.start()

        # Optional short wait so first subscribe happens quickly
        t0 = time.time()
        while (not self._user_connected.is_set() or not self._mkt_connected.is_set()) and (time.time() - t0 < 3.0):
            time.sleep(0.01)


    def _send_subscriptions(self) -> None:
        if not self._user_conn or not self._mkt_conn or not self._contract_id:
            return

        # Only send if both are connected (prevents errors during reconnect churn)
        if not self._user_connected.is_set() or not self._mkt_connected.is_set():
            return

        try:
            # User subscriptions
            self._user_conn.send("SubscribeAccounts", [])
            self._user_conn.send("SubscribeOrders", [self.account_id])
            self._user_conn.send("SubscribePositions", [self.account_id])
            self._user_conn.send("SubscribeTrades", [self.account_id])

            # Market subscriptions
            self._mkt_conn.send("SubscribeContractQuotes", [self._contract_id])
            self._mkt_conn.send("SubscribeContractTrades", [self._contract_id])
            if SUB_DEPTH:
                self._mkt_conn.send("SubscribeContractMarketDepth", [self._contract_id])

            self._log("Subscriptions sent:", f"account_id={self.account_id}", f"contract_id={self._contract_id}", f"depth={SUB_DEPTH}")
        except Exception as e:
            self._log("Subscription error:", repr(e))

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self._stop_evt.set()

        try:
            if self._user_conn:
                self._user_conn.stop()
        except Exception:
            pass
        try:
            if self._mkt_conn:
                self._mkt_conn.stop()
        except Exception:
            pass

    def snapshot(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        with self._lock:
            return list(self._state.orders.values()), list(self._state.positions.values())

    def last_quote(self, contract_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._state.last_quote_by_contract.get(str(contract_id))

    def last_trade(self, contract_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._state.last_trade_by_contract.get(str(contract_id))
