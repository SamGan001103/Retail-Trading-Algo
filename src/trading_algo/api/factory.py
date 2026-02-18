from __future__ import annotations

from trading_algo.api.client import ProjectXClient
from trading_algo.config import must_env


def client_from_env(token_ttl_sec: int = 25 * 60) -> ProjectXClient:
    return ProjectXClient(
        base_url=must_env("PROJECTX_BASE_URL"),
        username=must_env("PROJECTX_USERNAME"),
        api_key=must_env("PROJECTX_API_KEY"),
        token_ttl_sec=token_ttl_sec,
    )

