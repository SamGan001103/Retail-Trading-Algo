from __future__ import annotations

from trading_algo.broker import BrokerAdapter, broker_from_runtime_config
from trading_algo.config import RuntimeConfig, load_runtime_config


def load_runtime_and_broker() -> tuple[RuntimeConfig, BrokerAdapter]:
    config = load_runtime_config()
    broker = broker_from_runtime_config(config)
    return config, broker
