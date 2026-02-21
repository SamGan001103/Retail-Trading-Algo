from importlib import import_module
from typing import Any

__all__ = ["RealtimeClient", "main", "ModeOptions", "run_mode", "DrawdownGuard", "DrawdownSnapshot"]

_ATTR_TO_MODULE = {
    "RealtimeClient": "realtime_client",
    "main": "bot_runtime",
    "ModeOptions": "mode_runner",
    "run_mode": "mode_runner",
    "DrawdownGuard": "drawdown_guard",
    "DrawdownSnapshot": "drawdown_guard",
}


def __getattr__(name: str) -> Any:
    module_name = _ATTR_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value
