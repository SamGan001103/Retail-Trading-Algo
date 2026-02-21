from __future__ import annotations

import csv
import fnmatch
import json
import shutil
import tempfile
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

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


@dataclass(frozen=True)
class DatabentoImportStats:
    input_path: str
    output_csv: str
    files_processed: int
    rows_written: int
    skipped_rows: int
    source_files: list[str]
    start_utc: str
    end_utc: str


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


def _load_dbn_store_class() -> Any:
    # Prefer databento-dbn for offline DBN decoding; fallback to databento if present.
    modules: list[Any] = []
    try:
        import databento_dbn as dbn  # type: ignore[import-not-found]

        modules.append(dbn)
    except Exception:
        pass
    try:
        import databento as db  # type: ignore[import-not-found]

        modules.append(db)
    except Exception:
        pass
    for module in modules:
        store_cls = getattr(module, "DBNStore", None)
        if store_cls is not None:
            return store_cls
    raise RuntimeError(
        "Missing Databento DBN reader dependency for local .dbn/.dbn.zst decoding. "
        "DBN is Databento's default binary encoding and should be read via DBNStore. "
        "No API call is required to decode already-downloaded files. "
        "Install the Python SDK with `py -3.11 -m pip install databento`. "
        "If Python package install is blocked, transcode DBN to CSV externally and run "
        "`scripts/data/backtest_databento_daily.py --input <csv_dir_or_zip> ...`."
    )


def _is_dbn_filename(name: str) -> bool:
    lowered = name.strip().lower()
    return lowered.endswith(".dbn") or lowered.endswith(".dbn.zst")


def _matches(name: str, include: Sequence[str], exclude: Sequence[str]) -> bool:
    lowered = name.lower()
    if include:
        if not any(fnmatch.fnmatch(lowered, pattern.lower()) for pattern in include):
            return False
    if exclude:
        if any(fnmatch.fnmatch(lowered, pattern.lower()) for pattern in exclude):
            return False
    return True


def _discover_sources(
    input_path: Path,
    *,
    include: Sequence[str],
    exclude: Sequence[str],
    max_files: int | None,
) -> tuple[str, list[str]]:
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(input_path, "r") as archive:
            names = sorted(
                info.filename
                for info in archive.infolist()
                if (not info.is_dir()) and _is_dbn_filename(info.filename)
            )
        selected = [name for name in names if _matches(name, include, exclude)]
        if max_files is not None:
            selected = selected[: max(0, int(max_files))]
        return "zip", selected

    if input_path.is_dir():
        names = sorted(
            str(path)
            for path in input_path.rglob("*")
            if path.is_file() and _is_dbn_filename(str(path))
        )
        selected = [name for name in names if _matches(Path(name).name, include, exclude)]
        if max_files is not None:
            selected = selected[: max(0, int(max_files))]
        return "fs", selected

    if input_path.is_file() and _is_dbn_filename(input_path.name):
        if _matches(input_path.name, include, exclude):
            return "fs", [str(input_path)]
        return "fs", []

    raise RuntimeError(f"Unsupported input path for Databento import: {input_path}")


def _to_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_price(value: Any) -> float | None:
    raw = _to_number(value)
    if raw is None:
        return None
    # Databento raw DBN prices are int nanounits when pretty_px=False.
    if abs(raw) >= 1_000_000.0:
        return raw / 1_000_000_000.0
    return raw


def _to_iso_utc(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    if isinstance(value, (int, float)):
        numeric = int(value)
        abs_numeric = abs(numeric)
        if abs_numeric >= 1_000_000_000_000:
            sec, ns = divmod(numeric, 1_000_000_000)
            dt = datetime.fromtimestamp(sec, tz=timezone.utc).replace(microsecond=ns // 1000)
            return dt.isoformat().replace("+00:00", "Z")
        if abs_numeric >= 1_000_000_000:
            dt = datetime.fromtimestamp(float(numeric), tz=timezone.utc)
            return dt.isoformat().replace("+00:00", "Z")
        return None

    text = str(value).strip()
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
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _action_name(value: Any) -> str:
    if value is None:
        return ""
    enum_name = getattr(value, "name", None)
    if isinstance(enum_name, str) and enum_name.strip() != "":
        return enum_name.strip().lower()
    text = str(value).strip().lower()
    if "." in text:
        text = text.split(".")[-1].strip()
    return text


def _is_trade_action(name: str) -> bool:
    return name in {"trade", "t", "fill", "f", "last", "l"}


def _level_value(level: Any, primary: str, alt: str, index: int) -> Any:
    if isinstance(level, dict):
        if primary in level:
            return level[primary]
        if alt in level:
            return level[alt]
        return None
    if isinstance(level, (list, tuple)):
        if 0 <= index < len(level):
            return level[index]
        return None
    return getattr(level, primary, getattr(level, alt, None))


def _extract_depth_side(levels: Any, *, side: str) -> list[dict[str, float]]:
    if not isinstance(levels, (list, tuple)):
        return []
    out: list[dict[str, float]] = []
    for level in levels[:10]:
        if side == "bid":
            price = _to_price(_level_value(level, "bid_px", "bid_price", 0))
            size = _to_number(_level_value(level, "bid_sz", "bid_size", 1))
        else:
            price = _to_price(_level_value(level, "ask_px", "ask_price", 3))
            size = _to_number(_level_value(level, "ask_sz", "ask_size", 4))
        if price is None or size is None or size <= 0:
            continue
        out.append({"price": round(float(price), 10), "size": round(float(size), 10)})
    return out


def _record_to_orderflow_row(record: Any, *, fallback_seq: int, include_depth_json: bool) -> dict[str, Any] | None:
    ts = _to_iso_utc(getattr(record, "ts_event", None) or getattr(record, "ts_recv", None))
    if ts is None:
        return None

    depth_levels = getattr(record, "levels", None)
    bids = _extract_depth_side(depth_levels, side="bid")
    asks = _extract_depth_side(depth_levels, side="ask")

    best_bid = bids[0]["price"] if bids else None
    best_ask = asks[0]["price"] if asks else None
    best_bid_size = bids[0]["size"] if bids else None
    best_ask_size = asks[0]["size"] if asks else None

    action = _action_name(getattr(record, "action", None))
    raw_trade_price = _to_price(getattr(record, "price", None))
    raw_trade_size = _to_number(getattr(record, "size", None)) or 0.0
    is_trade = _is_trade_action(action)
    trade_price = raw_trade_price if (is_trade and raw_trade_price is not None) else None
    trade_size = raw_trade_size if is_trade else 0.0

    if best_bid is not None and best_ask is not None:
        price = (best_bid + best_ask) / 2.0
    elif trade_price is not None:
        price = trade_price
    else:
        price = raw_trade_price if raw_trade_price is not None else (best_bid if best_bid is not None else best_ask)
    if price is None:
        return None

    seq = _to_int(getattr(record, "sequence", None))
    if seq is None or seq <= 0:
        seq = fallback_seq

    return {
        "timestamp": ts,
        "seq": int(seq),
        "price": round(float(price), 10),
        "trade_price": "" if trade_price is None else round(float(trade_price), 10),
        "trade_size": round(float(trade_size), 10),
        "bid": "" if best_bid is None else round(float(best_bid), 10),
        "ask": "" if best_ask is None else round(float(best_ask), 10),
        "bestBid": "" if best_bid is None else round(float(best_bid), 10),
        "bestAsk": "" if best_ask is None else round(float(best_ask), 10),
        "bestBidSize": "" if best_bid_size is None else round(float(best_bid_size), 10),
        "bestAskSize": "" if best_ask_size is None else round(float(best_ask_size), 10),
        "depth_bids": json.dumps(bids, separators=(",", ":")) if include_depth_json else "",
        "depth_asks": json.dumps(asks, separators=(",", ":")) if include_depth_json else "",
    }


def _iter_store_records(store: Any) -> Iterator[Any]:
    # API surface differs slightly across databento-dbn versions.
    records_method = getattr(store, "records", None)
    if callable(records_method):
        try:
            yielded = records_method()
            if isinstance(yielded, Iterable):
                for record in yielded:
                    yield record
                return
        except TypeError:
            pass
    if isinstance(store, Iterable):
        for record in store:
            yield record


def _convert_single_dbn_file(
    *,
    dbn_path: Path,
    writer: csv.DictWriter,
    dbn_store_class: Any,
    include_depth_json: bool,
    next_fallback_seq: int,
    remaining_rows: int | None,
) -> tuple[int, int, int, bool]:
    store = dbn_store_class.from_file(str(dbn_path))
    written = 0
    skipped = 0
    stop = False
    try:
        for record in _iter_store_records(store):
            row = _record_to_orderflow_row(
                record,
                fallback_seq=next_fallback_seq,
                include_depth_json=include_depth_json,
            )
            next_fallback_seq += 1
            if row is None:
                skipped += 1
                continue
            writer.writerow(row)
            written += 1
            if remaining_rows is not None and written >= remaining_rows:
                stop = True
                break
    finally:
        closer = getattr(store, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass
    return written, skipped, next_fallback_seq, stop


def convert_databento_mbp10_to_orderflow_csv(
    *,
    input_path: str,
    output_csv: str,
    append: bool = False,
    include_depth_json: bool = True,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    max_files: int | None = None,
    max_rows: int | None = None,
) -> DatabentoImportStats:
    started = _now_iso_utc()
    include = [p.strip() for p in (include or []) if p.strip() != ""]
    exclude = [p.strip() for p in (exclude or []) if p.strip() != ""]
    input_path_obj = Path(input_path)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_type, selected_sources = _discover_sources(
        input_path_obj,
        include=include,
        exclude=exclude,
        max_files=max_files,
    )
    if not selected_sources:
        raise RuntimeError(
            "No Databento DBN sources selected. "
            "Check --include/--exclude patterns and input path."
        )

    dbn_store_class = _load_dbn_store_class()
    mode = "a" if append else "w"
    write_header = (not append) or (not output_path.exists()) or output_path.stat().st_size == 0
    next_fallback_seq = (_read_existing_max_seq(output_path) + 1) if append else 1
    rows_written = 0
    rows_skipped = 0
    files_processed = 0
    source_files: list[str] = []

    with output_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()

        if source_type == "zip":
            with zipfile.ZipFile(input_path_obj, "r") as archive, tempfile.TemporaryDirectory(
                prefix="databento_import_"
            ) as tmp_dir:
                tmp_root = Path(tmp_dir)
                for archive_name in selected_sources:
                    if max_rows is not None and max_rows > 0 and rows_written >= max_rows:
                        break
                    source_files.append(archive_name)
                    safe_name = archive_name.replace("/", "_").replace("\\", "_")
                    extracted = tmp_root / safe_name
                    with archive.open(archive_name, "r") as src, extracted.open("wb") as dst:
                        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
                    remaining = None
                    if max_rows is not None and max_rows > 0:
                        remaining = max(0, max_rows - rows_written)
                    written, skipped, next_fallback_seq, stop = _convert_single_dbn_file(
                        dbn_path=extracted,
                        writer=writer,
                        dbn_store_class=dbn_store_class,
                        include_depth_json=include_depth_json,
                        next_fallback_seq=next_fallback_seq,
                        remaining_rows=remaining,
                    )
                    rows_written += written
                    rows_skipped += skipped
                    files_processed += 1
                    try:
                        extracted.unlink()
                    except OSError:
                        pass
                    if stop:
                        break
        else:
            for source in selected_sources:
                if max_rows is not None and max_rows > 0 and rows_written >= max_rows:
                    break
                source_path = Path(source)
                source_files.append(str(source_path))
                remaining = None
                if max_rows is not None and max_rows > 0:
                    remaining = max(0, max_rows - rows_written)
                written, skipped, next_fallback_seq, stop = _convert_single_dbn_file(
                    dbn_path=source_path,
                    writer=writer,
                    dbn_store_class=dbn_store_class,
                    include_depth_json=include_depth_json,
                    next_fallback_seq=next_fallback_seq,
                    remaining_rows=remaining,
                )
                rows_written += written
                rows_skipped += skipped
                files_processed += 1
                if stop:
                    break

    finished = _now_iso_utc()
    return DatabentoImportStats(
        input_path=str(input_path_obj),
        output_csv=str(output_path),
        files_processed=files_processed,
        rows_written=rows_written,
        skipped_rows=rows_skipped,
        source_files=source_files,
        start_utc=started,
        end_utc=finished,
    )
