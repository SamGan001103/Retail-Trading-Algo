from __future__ import annotations

from trading_algo.api import ProjectXClient, client_from_env
from trading_algo.config import must_env
from trading_algo.execution.engine import ExecutionEngine


def engine_from_env() -> tuple[ProjectXClient, ExecutionEngine]:
    client = client_from_env()
    engine = ExecutionEngine(client, int(must_env("ACCOUNT_ID")))
    return client, engine

