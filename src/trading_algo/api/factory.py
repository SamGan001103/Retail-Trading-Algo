from __future__ import annotations

import os

from trading_algo.api.client import ProjectXClient


def _must_env_any(*names: str) -> str:
    for name in names:
        raw = (os.getenv(name) or "").strip().strip('"').strip("'")
        if raw:
            return raw
    keys = ", ".join(names)
    raise RuntimeError(f"Missing env var. Provide one of: {keys}")


def client_from_env(token_ttl_sec: int = 25 * 60) -> ProjectXClient:
    return ProjectXClient(
        base_url=_must_env_any("BROKER_BASE_URL", "PROJECTX_BASE_URL"),
        username=_must_env_any("BROKER_USERNAME", "PROJECTX_USERNAME"),
        api_key=_must_env_any("BROKER_API_KEY", "PROJECTX_API_KEY"),
        token_ttl_sec=token_ttl_sec,
    )

