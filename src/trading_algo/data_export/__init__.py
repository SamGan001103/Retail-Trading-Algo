__all__ = [
    "CaptureStats",
    "DatabentoImportStats",
    "capture_projectx_orderflow_csv",
    "convert_databento_mbp10_to_orderflow_csv",
]

from importlib import import_module
from typing import Any

_ATTR_TO_MODULE = {
    "CaptureStats": "projectx_orderflow",
    "capture_projectx_orderflow_csv": "projectx_orderflow",
    "DatabentoImportStats": "databento_orderflow",
    "convert_databento_mbp10_to_orderflow_csv": "databento_orderflow",
}


def __getattr__(name: str) -> Any:
    module_name = _ATTR_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value
