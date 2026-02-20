from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trading_algo.strategy import MarketBar


def _pick(row: dict[str, str], *keys: str) -> str:
    lowered = {k.lower(): v for k, v in row.items()}
    for key in keys:
        if key.lower() in lowered:
            return lowered[key.lower()]
    raise KeyError(f"Missing one of columns: {keys}")


def _pick_optional(row: dict[str, str], *keys: str) -> str | None:
    lowered = {k.lower(): v for k, v in row.items()}
    for key in keys:
        if key.lower() in lowered:
            value = lowered[key.lower()]
            if value is None:
                return None
            text = str(value).strip()
            if text == "":
                return None
            return text
    return None


def _num_optional(row: dict[str, str], *keys: str) -> float | None:
    raw = _pick_optional(row, *keys)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _json_list_optional(row: dict[str, str], *keys: str) -> list[dict[str, Any]] | None:
    raw = _pick_optional(row, *keys)
    if raw is None:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    out: list[dict[str, Any]] = []
    for item in parsed:
        if isinstance(item, dict):
            out.append(item)
    return out


@dataclass(frozen=True)
class OrderFlowTick:
    ts: str
    price: float
    volume: float
    quote: dict[str, Any] | None
    trade: dict[str, Any] | None
    depth: dict[str, Any] | None
    seq: int = 0


def load_bars_from_csv(path: str) -> list[MarketBar]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Backtest data file not found: {path}")

    bars: list[MarketBar] = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
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


def load_orderflow_ticks_from_csv(path: str) -> list[OrderFlowTick]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Backtest data file not found: {path}")

    ticks: list[OrderFlowTick] = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=1):
            ts = _pick(row, "timestamp", "datetime", "time", "date")
            bid = _num_optional(row, "bid", "bidPrice", "bestBid")
            ask = _num_optional(row, "ask", "askPrice", "bestAsk")
            trade_price = _num_optional(row, "trade_price", "tradePrice", "price", "last", "lastPrice", "close", "c")
            price = trade_price
            if price is None and bid is not None and ask is not None:
                price = (bid + ask) / 2.0
            if price is None:
                price = _num_optional(row, "close", "c")
            if price is None:
                raise RuntimeError(
                    f"Orderflow CSV row missing usable price at ts={ts!r}. Provide trade_price/price/close or bid+ask."
                )

            trade_size = _num_optional(row, "trade_size", "size", "qty", "quantity", "lastSize", "volume", "v")
            if trade_size is None:
                trade_size = 0.0

            quote: dict[str, Any] | None = None
            if bid is not None or ask is not None:
                quote = {}
                if bid is not None:
                    quote["bid"] = bid
                if ask is not None:
                    quote["ask"] = ask

            trade: dict[str, Any] | None = None
            if trade_price is not None or trade_size > 0:
                trade = {
                    "price": float(trade_price if trade_price is not None else price),
                    "size": float(trade_size),
                }

            best_bid_size = _num_optional(row, "bestBidSize", "bidSize", "bid_size")
            best_ask_size = _num_optional(row, "bestAskSize", "askSize", "ask_size")
            bids = _json_list_optional(row, "depth_bids", "bids", "book_bids")
            asks = _json_list_optional(row, "depth_asks", "asks", "book_asks")
            depth: dict[str, Any] | None = None
            if any(x is not None for x in (best_bid_size, best_ask_size, bids, asks, bid, ask)):
                depth = {}
                if best_bid_size is not None:
                    depth["bestBidSize"] = float(best_bid_size)
                if best_ask_size is not None:
                    depth["bestAskSize"] = float(best_ask_size)
                if bids is not None:
                    depth["bids"] = bids
                if asks is not None:
                    depth["asks"] = asks
                if bid is not None:
                    depth["bestBid"] = float(bid)
                if ask is not None:
                    depth["bestAsk"] = float(ask)

            ticks.append(
                OrderFlowTick(
                    ts=ts,
                    price=float(price),
                    volume=float(trade_size),
                    quote=quote,
                    trade=trade,
                    depth=depth,
                    seq=int(_num_optional(row, "seq", "sequence", "event_seq", "eventSequence") or row_idx),
                )
            )

    if not ticks:
        raise RuntimeError(f"No rows loaded from orderflow CSV: {path}")
    return ticks
