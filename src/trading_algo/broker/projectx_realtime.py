from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

from signalrcore.hub_connection_builder import HubConnectionBuilder
from signalrcore.protocol.json_hub_protocol import JsonHubProtocol

from trading_algo.api import ProjectXClient
from trading_algo.config import env_bool

DEBUG_RTC = env_bool("DEBUG_RTC", False)


@dataclass
class RTState:
    orders: dict[int, dict[str, Any]]
    positions: dict[int, dict[str, Any]]
    last_quote_by_contract: dict[str, dict[str, Any]]
    last_trade_by_contract: dict[str, dict[str, Any]]
    last_depth_by_contract: dict[str, dict[str, Any]]


class JsonHubProtocolRS(JsonHubProtocol):
    RS = "\x1e"

    def __init__(self):
        super().__init__()
        self._buffer = ""

    def parse_messages(self, raw: str):
        if not raw:
            return []
        self._buffer += raw
        parts = self._buffer.split(self.RS)
        self._buffer = parts.pop()
        out = []
        for part in parts:
            part = part.strip()
            if part:
                out.extend(super().parse_messages(part))
        if len(self._buffer) > 2_000_000:
            self._buffer = ""
        return out


class ProjectXRealtimeStream:
    def __init__(
        self,
        client: ProjectXClient,
        account_id: int,
        user_hub_url: str,
        market_hub_url: str,
    ):
        self.client = client
        self.account_id = int(account_id)
        self.user_hub_url = user_hub_url.rstrip("/")
        self.market_hub_url = market_hub_url.rstrip("/")

        self._state = RTState(
            orders={},
            positions={},
            last_quote_by_contract={},
            last_trade_by_contract={},
            last_depth_by_contract={},
        )
        self._lock = threading.RLock()
        self._running = False
        self._contract_id: str | None = None
        self._user_conn = None
        self._mkt_conn = None
        self._user_connected = threading.Event()
        self._mkt_connected = threading.Event()

    def _log(self, *args: Any) -> None:
        if DEBUG_RTC:
            print("[RTC]", *args)

    @staticmethod
    def _iter_dicts(value: Any):
        if isinstance(value, dict):
            yield value
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    yield item

    def _on_user_order(self, args: list[Any]) -> None:
        payload = args[0] if len(args) == 1 else args
        for item in self._iter_dicts(payload):
            order_id = item.get("id")
            if order_id is None:
                continue
            with self._lock:
                self._state.orders[int(order_id)] = item

    def _on_user_position(self, args: list[Any]) -> None:
        payload = args[0] if len(args) == 1 else args
        for item in self._iter_dicts(payload):
            position_id = item.get("id")
            if position_id is None:
                continue
            with self._lock:
                self._state.positions[int(position_id)] = item

    def _on_quote(self, args: list[Any]) -> None:
        if len(args) < 2 or not isinstance(args[1], dict):
            return
        with self._lock:
            self._state.last_quote_by_contract[str(args[0])] = args[1]

    def _on_trade(self, args: list[Any]) -> None:
        if len(args) < 2 or not isinstance(args[1], dict):
            return
        with self._lock:
            self._state.last_trade_by_contract[str(args[0])] = args[1]

    def _on_depth(self, args: list[Any]) -> None:
        if len(args) < 2 or not isinstance(args[1], dict):
            return
        with self._lock:
            self._state.last_depth_by_contract[str(args[0])] = args[1]

    def start(self, contract_id: str) -> None:
        if self._running:
            return
        self._running = True
        self._contract_id = str(contract_id)
        self._user_connected.clear()
        self._mkt_connected.clear()

        token = self.client.token()
        user_url = f"{self.user_hub_url}?access_token={token}"
        market_url = f"{self.market_hub_url}?access_token={token}"

        self._user_conn = (
            HubConnectionBuilder()
            .with_url(user_url)
            .with_hub_protocol(JsonHubProtocolRS())
            .with_automatic_reconnect(
                {"type": "raw", "keep_alive_interval": 10, "reconnect_interval": 5, "max_attempts": 999999}
            )
            .build()
        )
        self._mkt_conn = (
            HubConnectionBuilder()
            .with_url(market_url)
            .with_hub_protocol(JsonHubProtocolRS())
            .with_automatic_reconnect(
                {"type": "raw", "keep_alive_interval": 10, "reconnect_interval": 5, "max_attempts": 999999}
            )
            .build()
        )

        self._user_conn.on_open(
            lambda: (self._log("User hub connected"), self._user_connected.set(), self._send_subscriptions())
        )
        self._mkt_conn.on_open(
            lambda: (self._log("Market hub connected"), self._mkt_connected.set(), self._send_subscriptions())
        )
        self._user_conn.on_error(lambda data: self._log("User hub error:", data))
        self._mkt_conn.on_error(lambda data: self._log("Market hub error:", data))
        self._user_conn.on("GatewayUserOrder", self._on_user_order)
        self._user_conn.on("GatewayUserPosition", self._on_user_position)
        self._mkt_conn.on("GatewayQuote", self._on_quote)
        self._mkt_conn.on("GatewayTrade", self._on_trade)
        self._mkt_conn.on("GatewayMarketDepth", self._on_depth)
        self._user_conn.start()
        self._mkt_conn.start()

        start_ts = time.time()
        while (not self._user_connected.is_set() or not self._mkt_connected.is_set()) and (
            time.time() - start_ts < 3.0
        ):
            time.sleep(0.01)
        if not self._user_connected.is_set() or not self._mkt_connected.is_set():
            self.stop()
            raise RuntimeError("Realtime stream failed to connect (user/market hub) within 3 seconds.")

    def _send_subscriptions(self) -> None:
        if not self._user_conn or not self._mkt_conn or not self._contract_id:
            return
        if not self._user_connected.is_set() or not self._mkt_connected.is_set():
            return
        try:
            self._user_conn.send("SubscribeAccounts", [])
            self._user_conn.send("SubscribeOrders", [self.account_id])
            self._user_conn.send("SubscribePositions", [self.account_id])
            self._user_conn.send("SubscribeTrades", [self.account_id])
            self._mkt_conn.send("SubscribeContractQuotes", [self._contract_id])
            self._mkt_conn.send("SubscribeContractTrades", [self._contract_id])
            if env_bool("SUB_DEPTH", False):
                self._mkt_conn.send("SubscribeContractMarketDepth", [self._contract_id])
        except Exception as exc:
            self._log("Subscription error:", repr(exc))

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
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

    def snapshot(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        with self._lock:
            return list(self._state.orders.values()), list(self._state.positions.values())

    def last_quote(self, contract_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._state.last_quote_by_contract.get(str(contract_id))

    def last_trade(self, contract_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._state.last_trade_by_contract.get(str(contract_id))

    def last_depth(self, contract_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._state.last_depth_by_contract.get(str(contract_id))
