from __future__ import annotations

import os
from dataclasses import dataclass

from trading_algo.config.env import env_bool, env_int, must_env


@dataclass(frozen=True)
class RuntimeConfig:
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


def load_runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        base_url=must_env("PROJECTX_BASE_URL").rstrip("/"),
        username=must_env("PROJECTX_USERNAME"),
        api_key=must_env("PROJECTX_API_KEY"),
        account_id=int(must_env("ACCOUNT_ID")),
        user_hub_url=must_env("RTC_USER_HUB_URL"),
        market_hub_url=must_env("RTC_MARKET_HUB_URL"),
        symbol=(os.getenv("SYMBOL") or "MNQ").strip().upper(),
        live=env_bool("LIVE", False),
        side=env_int("SIDE", 0),
        size=env_int("SIZE", 1),
        sl_ticks=env_int("SL_TICKS", 40),
        tp_ticks=env_int("TP_TICKS", 80),
        loop_sec=env_int("LOOP_SEC", 1),
        exit_grace_sec=env_int("EXIT_GRACE_SEC", 5),
        flatten_on_start=env_bool("FLATTEN_ON_START", False),
        trade_on_start=env_bool("TRADE_ON_START", False),
    )
