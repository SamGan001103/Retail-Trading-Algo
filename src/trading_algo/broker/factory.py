from __future__ import annotations

from trading_algo.config.settings import RuntimeConfig

from .base import BrokerAdapter
from .projectx import ProjectXBrokerAdapter


def broker_from_runtime_config(config: RuntimeConfig) -> BrokerAdapter:
    broker = config.broker.strip().lower()
    if broker == "projectx":
        return ProjectXBrokerAdapter(
            base_url=config.base_url,
            username=config.username,
            api_key=config.api_key,
            user_hub_url=config.user_hub_url,
            market_hub_url=config.market_hub_url,
        )
    raise ValueError(f"Unsupported BROKER={config.broker!r}. Currently supported: projectx")
