from __future__ import annotations

import csv
from pathlib import Path

from trading_algo.strategy import MarketBar


def _pick(row: dict[str, str], *keys: str) -> str:
    lowered = {k.lower(): v for k, v in row.items()}
    for key in keys:
        if key.lower() in lowered:
            return lowered[key.lower()]
    raise KeyError(f"Missing one of columns: {keys}")


def load_bars_from_csv(path: str) -> list[MarketBar]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Backtest data file not found: {path}")

    bars: list[MarketBar] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bars.append(
                MarketBar(
                    ts=_pick(row, "timestamp", "datetime", "time", "date"),
                    open=float(_pick(row, "open", "o")),
                    high=float(_pick(row, "high", "h")),
                    low=float(_pick(row, "low", "l")),
                    close=float(_pick(row, "close", "c")),
                    volume=float(_pick(row, "volume", "v")),
                )
            )
    if not bars:
        raise RuntimeError(f"No rows loaded from backtest CSV: {path}")
    return bars

