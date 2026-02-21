from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
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
                continue
            text = str(value).strip()
            if text == "":
                continue
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


def _num_int_optional(row: dict[str, str], *keys: str) -> int | None:
    raw = _pick_optional(row, *keys)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _decode_databento_price(value: float | None) -> float | None:
    if value is None:
        return None
    # Databento raw CSV prices are often integer nanounits (pretty_px=False).
    if abs(value) >= 1_000_000.0:
        return value / 1_000_000_000.0
    return value


def _price_optional(row: dict[str, str], *keys: str) -> float | None:
    return _decode_databento_price(_num_optional(row, *keys))


def _normalize_timestamp(raw: str) -> str:
    text = str(raw).strip()
    if text == "":
        return text

    try:
        whole = int(text)
    except ValueError:
        whole = 0
    if whole != 0:
        abs_whole = abs(whole)
        # Heuristics for unix units (ns/us/ms/s).
        if abs_whole >= 100_000_000_000_000_000:  # ns
            sec, ns = divmod(whole, 1_000_000_000)
            dt = datetime.fromtimestamp(sec, tz=timezone.utc).replace(microsecond=ns // 1000)
            return dt.isoformat().replace("+00:00", "Z")
        if abs_whole >= 100_000_000_000_000:  # us
            sec, us = divmod(whole, 1_000_000)
            dt = datetime.fromtimestamp(sec, tz=timezone.utc).replace(microsecond=us)
            return dt.isoformat().replace("+00:00", "Z")
        if abs_whole >= 100_000_000_000:  # ms
            dt = datetime.fromtimestamp(whole / 1000.0, tz=timezone.utc)
            return dt.isoformat().replace("+00:00", "Z")
        if abs_whole >= 1_000_000_000:  # sec
            dt = datetime.fromtimestamp(float(whole), tz=timezone.utc)
            return dt.isoformat().replace("+00:00", "Z")

    iso_text = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        dt = datetime.fromisoformat(iso_text)
    except ValueError:
        return text
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_trade_action(action: str | None) -> bool:
    if action is None:
        return True
    text = str(action).strip().lower()
    return text in {"t", "trade", "fill", "f", "last", "l"}


def _databento_depth_levels(row: dict[str, str]) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    bids: list[dict[str, float]] = []
    asks: list[dict[str, float]] = []
    for level in range(10):
        suffix = f"{level:02d}"
        bid_px = _price_optional(row, f"bid_px_{suffix}", f"bid_px_{level}")
        bid_sz = _num_optional(row, f"bid_sz_{suffix}", f"bid_sz_{level}")
        ask_px = _price_optional(row, f"ask_px_{suffix}", f"ask_px_{level}")
        ask_sz = _num_optional(row, f"ask_sz_{suffix}", f"ask_sz_{level}")
        if bid_px is not None and bid_sz is not None and bid_sz > 0:
            bids.append({"price": float(bid_px), "size": float(bid_sz)})
        if ask_px is not None and ask_sz is not None and ask_sz > 0:
            asks.append({"price": float(ask_px), "size": float(ask_sz)})
    return bids, asks


def _looks_like_databento_row(row: dict[str, str]) -> bool:
    if _pick_optional(row, "ts_event", "ts_recv") is None:
        return False
    if _pick_optional(row, "bid_px_00", "ask_px_00") is not None:
        return True
    return _pick_optional(row, "sequence", "action") is not None


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
            ts = _normalize_timestamp(_pick(row, "timestamp", "datetime", "time", "date", "ts_event", "ts_recv"))
            databento_like = _looks_like_databento_row(row)
            action = _pick_optional(row, "action", "event_action", "record_action")
            is_trade = _is_trade_action(action) if databento_like else True

            bid = _price_optional(row, "bid", "bidPrice", "bestBid", "bid_px_00", "bid_px_0")
            ask = _price_optional(row, "ask", "askPrice", "bestAsk", "ask_px_00", "ask_px_0")
            raw_price = _price_optional(row, "trade_price", "tradePrice", "price", "last", "lastPrice", "close", "c")

            trade_price = raw_price if (raw_price is not None and is_trade) else None
            trade_size_raw = _num_optional(row, "trade_size", "size", "qty", "quantity", "lastSize", "volume", "v")
            trade_size = float(trade_size_raw or 0.0)
            if databento_like and not is_trade:
                trade_size = 0.0

            bids = _json_list_optional(row, "depth_bids", "bids", "book_bids")
            asks = _json_list_optional(row, "depth_asks", "asks", "book_asks")
            db_bids, db_asks = _databento_depth_levels(row) if databento_like else ([], [])
            if bids is None and db_bids:
                bids = db_bids
            if asks is None and db_asks:
                asks = db_asks

            if bid is None and bids:
                bid = _decode_databento_price(_num_optional(bids[0], "price"))
            if ask is None and asks:
                ask = _decode_databento_price(_num_optional(asks[0], "price"))

            if databento_like:
                if bid is not None and ask is not None:
                    price = (bid + ask) / 2.0
                elif trade_price is not None:
                    price = trade_price
                else:
                    price = raw_price
            else:
                price = trade_price
                if price is None:
                    price = raw_price
                if price is None and bid is not None and ask is not None:
                    price = (bid + ask) / 2.0
                if price is None:
                    price = _price_optional(row, "close", "c")
            if price is None:
                raise RuntimeError(
                    f"Orderflow CSV row missing usable price at ts={ts!r}. Provide trade_price/price/close or bid+ask."
                )

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

            best_bid_size = _num_optional(row, "bestBidSize", "bidSize", "bid_size", "bid_sz_00", "bid_sz_0")
            best_ask_size = _num_optional(row, "bestAskSize", "askSize", "ask_size", "ask_sz_00", "ask_sz_0")
            if best_bid_size is None and bids:
                best_bid_size = _num_optional(bids[0], "size")
            if best_ask_size is None and asks:
                best_ask_size = _num_optional(asks[0], "size")

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
                    seq=int(
                        _num_int_optional(
                            row,
                            "seq",
                            "sequence",
                            "event_seq",
                            "eventSequence",
                            "sequence_number",
                        )
                        or row_idx
                    ),
                )
            )

    if not ticks:
        raise RuntimeError(f"No rows loaded from orderflow CSV: {path}")
    return ticks
