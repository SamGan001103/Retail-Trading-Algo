"""Retail trading algo package."""

from importlib import import_module
from types import ModuleType

__all__ = [
    "api",
    "backtest",
    "broker",
    "config",
    "core",
    "data_export",
    "execution",
    "ml",
    "position_management",
    "runtime",
    "strategy",
    "telemetry",
]


def __getattr__(name: str) -> ModuleType:
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
