from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, time, timezone, tzinfo
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_TZ_FALLBACK_WARNED: set[str] = set()


def _pick(row: dict[str, Any], *keys: str) -> str:
    lowered = {k.lower(): v for k, v in row.items()}
    for key in keys:
        if key.lower() in lowered:
            return str(lowered[key.lower()])
    raise KeyError(f"Missing one of columns: {keys}")


def _pick_optional_value(row: dict[str, Any], *keys: str) -> Any | None:
    lowered = {k.lower(): v for k, v in row.items()}
    for key in keys:
        if key.lower() not in lowered:
            continue
        value = lowered[key.lower()]
        if value is None:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        return value
    return None


def _pick_optional(row: dict[str, Any], *keys: str) -> str | None:
    raw = _pick_optional_value(row, *keys)
    if raw is None:
        return None
    text = str(raw).strip()
    return text if text != "" else None


def _num_optional(row: dict[str, Any], *keys: str) -> float | None:
    raw = _pick_optional(row, *keys)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _num_int_optional(row: dict[str, Any], *keys: str) -> int | None:
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
    # Databento MBP prices may be integer nanounits when pretty_px=False.
    if abs(value) >= 1_000_000.0:
        return value / 1_000_000_000.0
    return value


def _price_optional(row: dict[str, Any], *keys: str) -> float | None:
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


def _parse_ts_utc(raw: str) -> datetime | None:
    text = str(raw).strip()
    if text == "":
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_clock(raw: str | None) -> time | None:
    text = str(raw or "").strip()
    if text == "":
        return None
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed.time()
        except ValueError:
            continue
    raise RuntimeError(f"Invalid session clock time: {text!r}. Expected HH:MM or HH:MM:SS.")


def _resolve_tzinfo(tz_name: str) -> tzinfo:
    key = str(tz_name or "").strip() or "UTC"
    try:
        return ZoneInfo(key)
    except (ZoneInfoNotFoundError, ValueError):
        if key not in _TZ_FALLBACK_WARNED:
            _TZ_FALLBACK_WARNED.add(key)
            print(
                f"timezone_warning source=backtest_data tz={key} resolved=UTC "
                "hint=install tzdata to ensure local-session filtering is accurate."
            )
        return timezone.utc


def _in_session_utc(
    dt_utc: datetime,
    *,
    start: time | None,
    end: time | None,
    tz: tzinfo,
    weekdays_only: bool,
) -> bool:
    dt_local = dt_utc.astimezone(tz)

    if weekdays_only and dt_local.weekday() >= 5:
        return False
    if start is None and end is None:
        return True
    if start is None or end is None:
        return True

    current = dt_local.time()
    if start <= end:
        return start <= current <= end
    return current >= start or current <= end


def _is_trade_action(action: str | None) -> bool:
    if action is None:
        return True
    text = str(action).strip().lower()
    return text in {"t", "trade", "fill", "f", "last", "l"}


def _databento_depth_levels(row: dict[str, Any]) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
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


def _looks_like_databento_row(row: dict[str, Any]) -> bool:
    if _pick_optional(row, "ts_event", "ts_recv") is None:
        return False
    if _pick_optional(row, "bid_px_00", "ask_px_00") is not None:
        return True
    return _pick_optional(row, "sequence", "action") is not None


def _json_list_optional(row: dict[str, Any], *keys: str) -> list[dict[str, Any]] | None:
    raw = _pick_optional_value(row, *keys)
    if raw is None:
        return None
    # Parquet-decoded nested lists may already be materialized as Python objects.
    if isinstance(raw, list):
        out: list[dict[str, Any]] = []
        for item in raw:
            if isinstance(item, dict):
                out.append(item)
        return out if out else None
    try:
        parsed = json.loads(str(raw))
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


def _build_orderflow_time_filter(
    *,
    session_start: str | None,
    session_end: str | None,
    tz_name: str,
    weekdays_only: bool,
) -> tuple[time | None, time | None, bool, tzinfo]:
    session_start_clock = _parse_clock(session_start)
    session_end_clock = _parse_clock(session_end)
    if (session_start_clock is None) != (session_end_clock is None):
        raise RuntimeError(
            "Orderflow session filter requires both session_start and session_end together."
        )
    apply_time_filter = session_start_clock is not None or session_end_clock is not None or bool(weekdays_only)
    tz_name_clean = str(tz_name or "America/New_York").strip() or "America/New_York"
    session_tz = _resolve_tzinfo(tz_name_clean)
    return session_start_clock, session_end_clock, apply_time_filter, session_tz


def _row_to_orderflow_tick(
    row: dict[str, Any],
    *,
    row_idx: int,
    apply_time_filter: bool,
    session_start_clock: time | None,
    session_end_clock: time | None,
    session_tz: tzinfo,
    weekdays_only: bool,
    start_utc: datetime | None = None,
    end_utc: datetime | None = None,
) -> OrderFlowTick | None:
    ts = _normalize_timestamp(_pick(row, "timestamp", "datetime", "time", "date", "ts_event", "ts_recv"))
    ts_dt = _parse_ts_utc(ts)
    if start_utc is not None or end_utc is not None:
        if ts_dt is None:
            return None
        if start_utc is not None and ts_dt < start_utc:
            return None
        if end_utc is not None and ts_dt >= end_utc:
            return None
    if apply_time_filter:
        if ts_dt is None:
            return None
        if not _in_session_utc(
            ts_dt,
            start=session_start_clock,
            end=session_end_clock,
            tz=session_tz,
            weekdays_only=bool(weekdays_only),
        ):
            return None

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
            f"Orderflow row missing usable price at ts={ts!r}. Provide trade_price/price/close or bid+ask."
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
    if bids is None and bid is not None and best_bid_size is not None:
        bids = [{"price": float(bid), "size": float(best_bid_size)}]
    if asks is None and ask is not None and best_ask_size is not None:
        asks = [{"price": float(ask), "size": float(best_ask_size)}]

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

    return OrderFlowTick(
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


def _load_pyarrow_parquet() -> Any:
    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - runtime dependency path
        raise RuntimeError(
            "Parquet backtest input requires `pyarrow`. Install with `py -3.11 -m pip install pyarrow`."
        ) from exc
    return pq


def _discover_parquet_sources(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() == ".parquet":
            return [path]
        raise RuntimeError(f"Expected .parquet file, got: {path}")
    if path.is_dir():
        files = sorted(p for p in path.rglob("*.parquet") if p.is_file())
        if files:
            return files
        raise RuntimeError(f"No .parquet files found in directory: {path}")
    raise FileNotFoundError(f"Backtest data file not found: {path}")


def is_orderflow_parquet_path(path: str) -> bool:
    path_obj = Path(path)
    if path_obj.is_file():
        return path_obj.suffix.lower() == ".parquet"
    if path_obj.is_dir():
        return any(path_obj.rglob("*.parquet"))
    return False


def iter_orderflow_ticks_from_parquet(
    path: str,
    *,
    session_start: str | None = None,
    session_end: str | None = None,
    tz_name: str = "America/New_York",
    weekdays_only: bool = False,
    max_rows: int | None = None,
    batch_size: int = 200_000,
    start_utc: datetime | None = None,
    end_utc: datetime | None = None,
) -> Iterator[OrderFlowTick]:
    sources = _discover_parquet_sources(Path(path))
    pq = _load_pyarrow_parquet()
    session_start_clock, session_end_clock, apply_time_filter, session_tz = _build_orderflow_time_filter(
        session_start=session_start,
        session_end=session_end,
        tz_name=tz_name,
        weekdays_only=weekdays_only,
    )
    row_cap = max(0, int(max_rows)) if max_rows is not None else 0
    emitted = 0
    row_idx = 0
    safe_batch = max(1, int(batch_size))

    for parquet_path in sources:
        pf = pq.ParquetFile(str(parquet_path))
        for batch in pf.iter_batches(batch_size=safe_batch):
            columns = batch.to_pydict()
            if not columns:
                continue
            names = list(columns.keys())
            first_name = names[0]
            rows = len(columns[first_name])
            for i in range(rows):
                row_idx += 1
                row = {name: columns[name][i] for name in names}
                tick = _row_to_orderflow_tick(
                    row,
                    row_idx=row_idx,
                    apply_time_filter=apply_time_filter,
                    session_start_clock=session_start_clock,
                    session_end_clock=session_end_clock,
                    session_tz=session_tz,
                    weekdays_only=bool(weekdays_only),
                    start_utc=start_utc,
                    end_utc=end_utc,
                )
                if tick is None:
                    continue
                yield tick
                emitted += 1
                if row_cap > 0 and emitted >= row_cap:
                    return


def load_orderflow_ticks_from_parquet(
    path: str,
    *,
    session_start: str | None = None,
    session_end: str | None = None,
    tz_name: str = "America/New_York",
    weekdays_only: bool = False,
    max_rows: int | None = None,
    batch_size: int = 200_000,
    start_utc: datetime | None = None,
    end_utc: datetime | None = None,
) -> list[OrderFlowTick]:
    ticks = list(
        iter_orderflow_ticks_from_parquet(
            path,
            session_start=session_start,
            session_end=session_end,
            tz_name=tz_name,
            weekdays_only=weekdays_only,
            max_rows=max_rows,
            batch_size=batch_size,
            start_utc=start_utc,
            end_utc=end_utc,
        )
    )
    if not ticks:
        raise RuntimeError(f"No rows loaded from orderflow Parquet: {path}")
    return ticks


def orderflow_tick_has_usable_depth(tick: OrderFlowTick) -> bool:
    depth = tick.depth
    if not isinstance(depth, dict):
        return False
    bids = depth.get("bids")
    asks = depth.get("asks")
    if isinstance(bids, list) and len(bids) > 0:
        return True
    if isinstance(asks, list) and len(asks) > 0:
        return True
    for key in ("bestBidSize", "bidSize", "bestAskSize", "askSize"):
        value = depth.get(key)
        if isinstance(value, (int, float)) and float(value) > 0:
            return True
    return False


def _tick_has_quote(tick: OrderFlowTick) -> bool:
    quote_bid = _num_optional(tick.quote or {}, "bid", "bidPrice", "bestBid")
    quote_ask = _num_optional(tick.quote or {}, "ask", "askPrice", "bestAsk")
    depth_bid = _num_optional(tick.depth or {}, "bestBid", "bidPrice")
    depth_ask = _num_optional(tick.depth or {}, "bestAsk", "askPrice")
    return (quote_bid is not None and quote_ask is not None) or (depth_bid is not None and depth_ask is not None)


def _timestamp_has_explicit_tz(raw: str) -> bool:
    text = str(raw).strip()
    if text.endswith("Z") or text.endswith("z"):
        return True
    return len(text) >= 6 and (text[-6] in {"+", "-"} and text[-3] == ":")


@dataclass(frozen=True)
class OrderFlowParquetScan:
    path: str
    source_files: list[str]
    columns: list[str]
    rows: int
    parseable_timestamps: int
    explicit_tz_timestamps: int
    quote_rows: int
    depth_rows: int
    first_ts_utc: datetime | None
    last_ts_utc: datetime | None


def scan_orderflow_parquet(
    path: str,
    *,
    session_start: str | None = None,
    session_end: str | None = None,
    tz_name: str = "America/New_York",
    weekdays_only: bool = False,
    max_rows: int | None = None,
    batch_size: int = 200_000,
    start_utc: datetime | None = None,
    end_utc: datetime | None = None,
) -> OrderFlowParquetScan:
    sources = _discover_parquet_sources(Path(path))
    pq = _load_pyarrow_parquet()
    headers: set[str] = set()
    for source in sources:
        schema_names = list(pq.ParquetFile(str(source)).schema_arrow.names)
        headers.update(schema_names)

    rows = 0
    parseable_timestamps = 0
    explicit_tz_timestamps = 0
    quote_rows = 0
    depth_rows = 0
    first_ts: datetime | None = None
    last_ts: datetime | None = None

    for tick in iter_orderflow_ticks_from_parquet(
        path,
        session_start=session_start,
        session_end=session_end,
        tz_name=tz_name,
        weekdays_only=weekdays_only,
        max_rows=max_rows,
        batch_size=batch_size,
        start_utc=start_utc,
        end_utc=end_utc,
    ):
        rows += 1
        if _timestamp_has_explicit_tz(tick.ts):
            explicit_tz_timestamps += 1
        ts_dt = _parse_ts_utc(tick.ts)
        if ts_dt is not None:
            parseable_timestamps += 1
            if first_ts is None or ts_dt < first_ts:
                first_ts = ts_dt
            if last_ts is None or ts_dt > last_ts:
                last_ts = ts_dt
        if _tick_has_quote(tick):
            quote_rows += 1
        if orderflow_tick_has_usable_depth(tick):
            depth_rows += 1

    return OrderFlowParquetScan(
        path=path,
        source_files=[str(p) for p in sources],
        columns=sorted(headers),
        rows=rows,
        parseable_timestamps=parseable_timestamps,
        explicit_tz_timestamps=explicit_tz_timestamps,
        quote_rows=quote_rows,
        depth_rows=depth_rows,
        first_ts_utc=first_ts,
        last_ts_utc=last_ts,
    )
