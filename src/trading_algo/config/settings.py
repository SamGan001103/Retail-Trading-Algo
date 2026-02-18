from __future__ import annotations

import os
from dataclasses import dataclass

from trading_algo.config.env import env_bool, env_int
from trading_algo.core import BUY, SELL


@dataclass(frozen=True)
class RuntimeConfig:
    broker: str
    base_url: str
    username: str
    api_key: str
    account_id: int
    user_hub_url: str
    market_hub_url: str
    symbol: str
    live: bool
    side: int
    size: int
    sl_ticks: int
    tp_ticks: int
    loop_sec: int
    exit_grace_sec: int
    flatten_on_start: bool
    trade_on_start: bool


def _must_env_any(*names: str) -> str:
    for name in names:
        raw = (os.getenv(name) or "").strip().strip('"').strip("'")
        if raw:
            return raw
    keys = ", ".join(names)
    raise RuntimeError(f"Missing env var. Provide one of: {keys}")


def _parse_side(raw: str | None) -> int:
    value = (raw or "").strip().lower()
    if not value:
        return BUY
    if value in {"0", "buy", "long"}:
        return BUY
    if value in {"1", "sell", "short"}:
        return SELL
    raise ValueError("Invalid SIDE value. Use 0/1 or buy/sell or long/short.")


def load_runtime_config() -> RuntimeConfig:
    broker = (os.getenv("BROKER") or "projectx").strip().lower()
    return RuntimeConfig(
        broker=broker,
        base_url=_must_env_any("BROKER_BASE_URL", "PROJECTX_BASE_URL").rstrip("/"),
        username=_must_env_any("BROKER_USERNAME", "PROJECTX_USERNAME"),
        api_key=_must_env_any("BROKER_API_KEY", "PROJECTX_API_KEY"),
        account_id=int(_must_env_any("ACCOUNT_ID")),
        user_hub_url=_must_env_any("BROKER_USER_HUB_URL", "RTC_USER_HUB_URL"),
        market_hub_url=_must_env_any("BROKER_MARKET_HUB_URL", "RTC_MARKET_HUB_URL"),
        symbol=(os.getenv("SYMBOL") or "MNQ").strip().upper(),
        live=env_bool("LIVE", False),
        side=_parse_side(os.getenv("SIDE")),
        size=env_int("SIZE", 1),
        sl_ticks=env_int("SL_TICKS", 40),
        tp_ticks=env_int("TP_TICKS", 80),
        loop_sec=env_int("LOOP_SEC", 1),
        exit_grace_sec=env_int("EXIT_GRACE_SEC", 5),
        flatten_on_start=env_bool("FLATTEN_ON_START", False),
        trade_on_start=env_bool("TRADE_ON_START", False),
    )
