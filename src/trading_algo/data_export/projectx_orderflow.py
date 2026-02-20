from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading_algo.broker import broker_from_runtime_config
from trading_algo.config import load_runtime_config

CSV_COLUMNS = [
    "timestamp",
    "seq",
    "price",
    "trade_price",
    "trade_size",
    "bid",
    "ask",
    "bestBid",
    "bestAsk",
    "bestBidSize",
    "bestAskSize",
    "depth_bids",
    "depth_asks",
]

_TS_KEYS = (
    "timestamp",
    "time",
    "ts",
    "dateTime",
    "datetime",
    "eventTime",
    "tradeTime",
    "quoteTime",
    "updateTime",
    "updatedAt",
)


@dataclass(frozen=True)
class CaptureStats:
    output_csv: str
    contract_id: str
    rows_written: int
    skipped_no_price: int
    skipped_no_depth: int
    duplicates_skipped: int
    duration_sec: float
    start_utc: str
    end_utc: str


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _num(payload: dict[str, Any] | None, *keys: str) -> float | None:
    if payload is None:
        return None
    lowered = {str(k).strip().lower(): v for k, v in payload.items()}
    for key in keys:
        value = lowered.get(key.lower())
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        as_float = float(value)
    else:
        text = str(value).strip()
        if text == "":
            return None
        try:
            as_float = float(text)
        except ValueError:
            as_float = float("nan")
        if as_float != as_float:
            raw = text
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            try:
                dt = datetime.fromisoformat(raw)
            except ValueError:
                return None
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)

    if as_float > 1e12:
        return datetime.fromtimestamp(as_float / 1000.0, tz=timezone.utc)
    if as_float > 1e9:
        return datetime.fromtimestamp(as_float, tz=timezone.utc)
    return None


def _extract_payload_ts(payload: dict[str, Any] | None) -> datetime | None:
    if not isinstance(payload, dict):
        return None
    for key in _TS_KEYS:
        if key in payload:
            parsed = _parse_timestamp(payload.get(key))
            if parsed is not None:
                return parsed
    return None


def _normalize_depth_levels(levels: Any) -> list[dict[str, float]]:
    if not isinstance(levels, list):
        return []
    out: list[dict[str, float]] = []
    for level in levels:
        if isinstance(level, dict):
            px = _num(level, "price", "p")
            sz = _num(level, "size", "qty", "q", "volume")
            if px is None or sz is None:
                continue
            out.append({"price": float(px), "size": float(sz)})
            continue
        if isinstance(level, (list, tuple)) and len(level) >= 2:
            try:
                out.append({"price": float(level[0]), "size": float(level[1])})
            except (TypeError, ValueError):
                continue
    return out


def _best_levels_from_depth(depth: dict[str, Any] | None) -> tuple[float | None, float | None, float | None, float | None]:
    if not isinstance(depth, dict):
        return None, None, None, None
    best_bid = _num(depth, "bestBid", "bidPrice")
    best_ask = _num(depth, "bestAsk", "askPrice")
    best_bid_size = _num(depth, "bestBidSize", "bidSize", "bid_size")
    best_ask_size = _num(depth, "bestAskSize", "askSize", "ask_size")
    return best_bid, best_ask, best_bid_size, best_ask_size


def _has_depth_signal(depth: dict[str, Any] | None) -> bool:
    if not isinstance(depth, dict):
        return False
    bids = _normalize_depth_levels(depth.get("bids"))
    asks = _normalize_depth_levels(depth.get("asks"))
    if bids or asks:
        return True
    if _num(depth, "bestBidSize", "bidSize", "bid_size") is not None:
        return True
    if _num(depth, "bestAskSize", "askSize", "ask_size") is not None:
        return True
    return False


def _row_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("price"),
        row.get("trade_price"),
        row.get("trade_size"),
        row.get("bid"),
        row.get("ask"),
        row.get("bestBidSize"),
        row.get("bestAskSize"),
        row.get("depth_bids"),
        row.get("depth_asks"),
    )


def _read_existing_max_seq(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    max_seq = 0
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw = str(row.get("seq") or "").strip()
                if raw == "":
                    continue
                try:
                    max_seq = max(max_seq, int(raw))
                except ValueError:
                    continue
    except OSError:
        return 0
    return max_seq


def build_orderflow_row(
    *,
    seq: int,
    quote: dict[str, Any] | None,
    trade: dict[str, Any] | None,
    depth: dict[str, Any] | None,
    now_utc: datetime,
    include_depth_json: bool,
    require_depth: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    if require_depth and not _has_depth_signal(depth):
        return None, "no-depth"

    bid = _num(quote, "bid", "bidPrice", "bestBid")
    ask = _num(quote, "ask", "askPrice", "bestAsk")
    best_bid, best_ask, best_bid_size, best_ask_size = _best_levels_from_depth(depth)
    if bid is None:
        bid = best_bid
    if ask is None:
        ask = best_ask

    trade_price = _num(trade, "price", "last", "lastPrice", "tradePrice", "close")
    trade_size = _num(trade, "size", "qty", "quantity", "volume", "lastSize") or 0.0

    price = trade_price
    if price is None and bid is not None and ask is not None:
        price = (float(bid) + float(ask)) / 2.0
    if price is None:
        price = bid if bid is not None else ask
    if price is None:
        return None, "no-price"

    ts = _extract_payload_ts(trade) or _extract_payload_ts(quote) or _extract_payload_ts(depth) or now_utc
    ts_iso = ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    bids = _normalize_depth_levels(depth.get("bids") if isinstance(depth, dict) else None)
    asks = _normalize_depth_levels(depth.get("asks") if isinstance(depth, dict) else None)
    if not bids and best_bid is not None and best_bid_size is not None:
        bids = [{"price": float(best_bid), "size": float(best_bid_size)}]
    if not asks and best_ask is not None and best_ask_size is not None:
        asks = [{"price": float(best_ask), "size": float(best_ask_size)}]

    row: dict[str, Any] = {
        "timestamp": ts_iso,
        "seq": int(seq),
        "price": round(float(price), 10),
        "trade_price": "" if trade_price is None else round(float(trade_price), 10),
        "trade_size": round(float(trade_size), 10),
        "bid": "" if bid is None else round(float(bid), 10),
        "ask": "" if ask is None else round(float(ask), 10),
        "bestBid": "" if best_bid is None else round(float(best_bid), 10),
        "bestAsk": "" if best_ask is None else round(float(best_ask), 10),
        "bestBidSize": "" if best_bid_size is None else round(float(best_bid_size), 10),
        "bestAskSize": "" if best_ask_size is None else round(float(best_ask_size), 10),
        "depth_bids": json.dumps(bids, separators=(",", ":")) if include_depth_json else "",
        "depth_asks": json.dumps(asks, separators=(",", ":")) if include_depth_json else "",
    }
    return row, None


def capture_projectx_orderflow_csv(
    *,
    output_csv: str,
    symbol: str,
    live: bool,
    duration_sec: float,
    poll_sec: float = 0.05,
    max_rows: int | None = None,
    append: bool = False,
    include_depth_json: bool = True,
    require_depth: bool = True,
    heartbeat_sec: float = 5.0,
) -> CaptureStats:
    config = load_runtime_config()
    broker = broker_from_runtime_config(config)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    skipped_no_price = 0
    skipped_no_depth = 0
    duplicates_skipped = 0
    seq = (_read_existing_max_seq(output_path) + 1) if append else 1
    mode = "a" if append else "w"

    start_utc = _now_iso_utc()
    started = time.time()
    next_heartbeat = started + max(0.5, float(heartbeat_sec))
    last_sig: tuple[Any, ...] | None = None

    contract_id = ""
    stream = None
    try:
        contract_id = broker.resolve_contract_id(symbol, live)
        stream = broker.create_stream(config.account_id)
        stream.start(contract_id)

        with output_path.open(mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if not append or output_path.stat().st_size == 0:
                writer.writeheader()

            while True:
                now = time.time()
                elapsed = now - started
                if duration_sec > 0 and elapsed >= duration_sec:
                    break
                if max_rows is not None and max_rows > 0 and rows_written >= max_rows:
                    break

                quote = stream.last_quote(contract_id) if stream is not None else None
                trade = stream.last_trade(contract_id) if stream is not None else None
                depth = stream.last_depth(contract_id) if stream is not None else None
                row, reason = build_orderflow_row(
                    seq=seq,
                    quote=quote,
                    trade=trade,
                    depth=depth,
                    now_utc=datetime.now(timezone.utc),
                    include_depth_json=include_depth_json,
                    require_depth=require_depth,
                )
                if row is None:
                    if reason == "no-depth":
                        skipped_no_depth += 1
                    elif reason == "no-price":
                        skipped_no_price += 1
                    time.sleep(max(0.01, float(poll_sec)))
                    continue

                signature = _row_signature(row)
                if signature == last_sig:
                    duplicates_skipped += 1
                    time.sleep(max(0.01, float(poll_sec)))
                    continue

                writer.writerow(row)
                f.flush()
                rows_written += 1
                seq += 1
                last_sig = signature

                if now >= next_heartbeat:
                    print(
                        f"capture rows={rows_written} elapsed_sec={elapsed:.1f} "
                        f"skipped_no_depth={skipped_no_depth} dup={duplicates_skipped}"
                    )
                    next_heartbeat = now + max(0.5, float(heartbeat_sec))

                time.sleep(max(0.01, float(poll_sec)))
    finally:
        try:
            if stream is not None:
                stream.stop()
        except Exception:
            pass
        broker.close()

    end_utc = _now_iso_utc()
    return CaptureStats(
        output_csv=str(output_path),
        contract_id=contract_id,
        rows_written=rows_written,
        skipped_no_price=skipped_no_price,
        skipped_no_depth=skipped_no_depth,
        duplicates_skipped=duplicates_skipped,
        duration_sec=max(0.0, time.time() - started),
        start_utc=start_utc,
        end_utc=end_utc,
    )
