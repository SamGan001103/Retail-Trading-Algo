from __future__ import annotations

import calendar
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from trading_algo.backtest import (
    BacktestConfig,
    OrderFlowParquetScan,
    OrderFlowTick,
    is_orderflow_parquet_path,
    iter_orderflow_ticks_from_parquet,
    orderflow_tick_has_usable_depth,
    run_backtest_orderflow,
    scan_orderflow_parquet,
)
from trading_algo.config import RuntimeConfig, env_bool, env_float, env_int, get_symbol_profile, load_runtime_config, must_env
from trading_algo.core import BUY, SELL
from trading_algo.ml import SetupMLGate, train_xgboost_from_parquet
from trading_algo.strategy import MarketBar, NYSessionMarketStructureStrategy, OneShotLongStrategy
from trading_algo.telemetry import TelemetryRouter


@dataclass(frozen=True)
class ModeOptions:
    mode: str
    data_path: str | None
    strategy: str
    model_out: str
    hold_bars: int
    profile: str | None = None


@dataclass(frozen=True)
class _WindowSpec:
    window_id: str
    start_utc: datetime
    end_utc: datetime
    months: int


@dataclass(frozen=True)
class _RunScenario:
    scenario_id: str
    config: BacktestConfig


_TZ_FALLBACK_WARNED: set[str] = set()


_BACKTEST_CANDIDATE_COLUMNS = [
    "run_id",
    "strategy",
    "scenario_id",
    "window_id",
    "window_start_utc",
    "window_end_utc",
    "window_months",
    "event_name",
    "bar_index",
    "bar_ts",
    "candidate_id",
    "status",
    "reason",
    "side",
    "setup_index",
    "setup_ts",
    "setup_close",
    "has_recent_sweep",
    "htf_bias",
    "bias_ok",
    "continuation",
    "reversal",
    "equal_levels",
    "fib_retracement",
    "key_area_proximity",
    "confluence_score",
    "of_imbalance",
    "of_top_bid_size",
    "of_top_ask_size",
    "of_best_bid",
    "of_best_ask",
    "of_spread",
    "of_trade_size",
    "of_trade_price",
    "ml_shadow_score",
    "ml_shadow_reason",
    "ml_shadow_approved",
    "source",
]

_BACKTEST_MATRIX_COLUMNS = [
    "run_id",
    "strategy",
    "scenario_id",
    "window_id",
    "window_start_utc",
    "window_end_utc",
    "window_months",
    "candidate_id",
    "side",
    "setup_index",
    "setup_ts",
    "setup_close",
    "has_recent_sweep",
    "htf_bias",
    "bias_ok",
    "continuation",
    "reversal",
    "equal_levels",
    "fib_retracement",
    "key_area_proximity",
    "confluence_score",
    "of_imbalance",
    "of_top_bid_size",
    "of_top_ask_size",
    "of_best_bid",
    "of_best_ask",
    "of_spread",
    "of_trade_size",
    "of_trade_price",
    "ml_shadow_score",
    "ml_shadow_reason",
    "ml_shadow_approved",
    "status_final",
    "reason_final",
    "entry_ts",
    "entry_price",
    "entry_event",
    "entry_reason",
    "entry_bar_index",
    "entry_order_id",
    "entry_position_id",
    "size",
    "sl_ticks_abs",
    "tp_ticks_abs",
    "risk_dollars",
    "exit_ts",
    "exit_price",
    "exit_event",
    "exit_reason",
    "exit_bar_index",
    "exit_order_id",
    "exit_position_id",
    "entry_count",
    "exit_count",
    "gross_entry_size",
    "gross_exit_size",
    "pnl",
    "win",
    "loss",
    "realized_rr",
    "result_label",
]

_BACKTEST_SUMMARY_COLUMNS = [
    "run_id",
    "strategy",
    "source_path",
    "scenario_id",
    "window_id",
    "window_start_utc",
    "window_end_utc",
    "window_months",
    "bars",
    "num_trades",
    "final_equity",
    "net_pnl",
    "return_pct",
    "win_rate_pct",
    "max_drawdown_pct",
    "orderflow_replay",
    "news_blackouts",
]


def _resolve_profile(explicit: str | None) -> str:
    raw = (explicit or os.getenv("RUNTIME_PROFILE") or "normal").strip().lower()
    if raw in {"debug", "dbg"}:
        return "debug"
    return "normal"


def _parse_ts_utc(ts: str) -> datetime | None:
    raw = str(ts).strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_utc_boundary(raw: str | None, *, end_of_day: bool = False) -> datetime | None:
    text = (raw or "").strip()
    if text == "":
        return None
    if "T" not in text:
        suffix = "T23:59:59.999999Z" if end_of_day else "T00:00:00Z"
        text = f"{text}{suffix}"
    parsed = _parse_ts_utc(text)
    return parsed


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _subtract_months(dt: datetime, months: int) -> datetime:
    total_months = dt.year * 12 + (dt.month - 1) - months
    year = total_months // 12
    month = total_months % 12 + 1
    last_day = calendar.monthrange(year, month)[1]
    day = min(dt.day, last_day)
    return dt.replace(year=year, month=month, day=day)


def _add_months(dt: datetime, months: int) -> datetime:
    total_months = dt.year * 12 + (dt.month - 1) + months
    year = total_months // 12
    month = total_months % 12 + 1
    last_day = calendar.monthrange(year, month)[1]
    day = min(dt.day, last_day)
    return dt.replace(year=year, month=month, day=day)


def _latest_months_window(
    bars: list[MarketBar],
    months: int,
) -> tuple[list[MarketBar], datetime | None, datetime | None]:
    if not bars:
        return bars, None, None
    safe_months = max(1, int(months))
    parsed: list[tuple[MarketBar, datetime]] = []
    for bar in bars:
        dt = _parse_ts_utc(bar.ts)
        if dt is not None:
            parsed.append((bar, dt))
    if not parsed:
        return bars, None, None

    end_dt = max(dt for _, dt in parsed)
    start_dt = _subtract_months(end_dt, safe_months)
    selected = [bar for bar, dt in parsed if start_dt <= dt <= end_dt]
    if not selected:
        return bars, start_dt, end_dt
    return selected, start_dt, end_dt


def _latest_months_window_ticks(
    ticks: list[OrderFlowTick],
    months: int,
) -> tuple[list[OrderFlowTick], datetime | None, datetime | None]:
    if not ticks:
        return ticks, None, None
    safe_months = max(1, int(months))
    parsed: list[tuple[OrderFlowTick, datetime]] = []
    for tick in ticks:
        dt = _parse_ts_utc(tick.ts)
        if dt is not None:
            parsed.append((tick, dt))
    if not parsed:
        return ticks, None, None

    end_dt = max(dt for _, dt in parsed)
    start_dt = _subtract_months(end_dt, safe_months)
    selected = [tick for tick, dt in parsed if start_dt <= dt <= end_dt]
    if not selected:
        return ticks, start_dt, end_dt
    return selected, start_dt, end_dt


def _parsed_bars_utc(bars: list[MarketBar]) -> list[tuple[MarketBar, datetime]]:
    parsed: list[tuple[MarketBar, datetime]] = []
    for bar in bars:
        dt = _parse_ts_utc(bar.ts)
        if dt is not None:
            parsed.append((bar, dt))
    return parsed


def _parsed_ticks_utc(ticks: list[OrderFlowTick]) -> list[tuple[OrderFlowTick, datetime]]:
    parsed: list[tuple[OrderFlowTick, datetime]] = []
    for tick in ticks:
        dt = _parse_ts_utc(tick.ts)
        if dt is not None:
            parsed.append((tick, dt))
    return parsed


def _build_walk_forward_windows(
    *,
    parsed_times_utc: list[datetime],
    window_months: int,
    step_months: int,
    start_utc: datetime | None,
    end_utc: datetime | None,
) -> list[_WindowSpec]:
    if not parsed_times_utc:
        return []
    safe_window = max(1, int(window_months))
    safe_step = max(1, int(step_months))
    data_start = min(parsed_times_utc)
    data_end = max(parsed_times_utc)
    range_start = max(data_start, start_utc) if start_utc is not None else data_start
    range_end = min(data_end, end_utc) if end_utc is not None else data_end
    if range_end <= range_start:
        return []

    windows: list[_WindowSpec] = []
    cursor = range_start
    idx = 1
    while cursor < range_end:
        window_end = _add_months(cursor, safe_window)
        if window_end > range_end:
            break
        windows.append(
            _WindowSpec(
                window_id=f"wf_{idx:03d}",
                start_utc=cursor,
                end_utc=window_end,
                months=safe_window,
            )
        )
        idx += 1
        next_cursor = _add_months(cursor, safe_step)
        if next_cursor <= cursor:
            break
        cursor = next_cursor
    return windows


def _bars_in_window(parsed_bars: list[tuple[MarketBar, datetime]], window: _WindowSpec) -> list[MarketBar]:
    return [bar for bar, dt in parsed_bars if window.start_utc <= dt < window.end_utc]


def _ticks_in_window(parsed_ticks: list[tuple[OrderFlowTick, datetime]], window: _WindowSpec) -> list[OrderFlowTick]:
    return [tick for tick, dt in parsed_ticks if window.start_utc <= dt < window.end_utc]


def _tick_has_usable_orderflow(tick: OrderFlowTick) -> bool:
    return orderflow_tick_has_usable_depth(tick)


def _ensure_orderflow_backtest_dataset(ticks: list[OrderFlowTick], path: str) -> None:
    if any(_tick_has_usable_orderflow(tick) for tick in ticks):
        return
    raise RuntimeError(
        "Backtest orderflow replay requires depth-capable rows. "
        f"No usable depth data found in source rows: {path}. "
        "Provide bestBidSize/bestAskSize or depth_bids/depth_asks JSON columns."
    )


def _has_any_column(columns: list[str], aliases: tuple[str, ...]) -> bool:
    lowered = {str(col).strip().lower() for col in columns}
    return any(alias.lower() in lowered for alias in aliases)


def _parse_hhmm_optional(raw: str, fallback: str) -> time:
    value = (raw or fallback).strip()
    parts = value.split(":")
    if len(parts) != 2:
        return datetime.strptime(fallback, "%H:%M").time()
    try:
        hour = int(parts[0])
        minute = int(parts[1])
    except ValueError:
        return datetime.strptime(fallback, "%H:%M").time()
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return datetime.strptime(fallback, "%H:%M").time()
    return time(hour=hour, minute=minute)


def _resolve_tzinfo(tz_name: str) -> tzinfo:
    key = str(tz_name or "").strip() or "UTC"
    try:
        return ZoneInfo(key)
    except (ZoneInfoNotFoundError, ValueError):
        if key not in _TZ_FALLBACK_WARNED:
            _TZ_FALLBACK_WARNED.add(key)
            print(
                f"timezone_warning source=mode_runner tz={key} resolved=UTC "
                "hint=install tzdata to ensure local-session filtering is accurate."
            )
        return timezone.utc


def _in_session_utc(
    dt_utc: datetime,
    *,
    start: time,
    end: time,
    tz_name: str | tzinfo,
    weekdays_only: bool = True,
) -> bool:
    tz = _resolve_tzinfo(tz_name) if isinstance(tz_name, str) else tz_name
    dt_local = dt_utc.astimezone(tz)
    if weekdays_only and dt_local.weekday() >= 5:
        return False
    t = dt_local.time()
    if start <= end:
        return start <= t < end
    return t >= start or t < end


def _timestamp_has_explicit_tz(raw: str) -> bool:
    text = str(raw).strip()
    if text.endswith("Z") or text.endswith("z"):
        return True
    if len(text) >= 6 and (text[-6] in {"+", "-"} and text[-3] == ":"):
        return True
    return False


def _validate_orderflow_backtest_preflight_parquet(
    *,
    data_path: str,
    scan: OrderFlowParquetScan,
    window_start: datetime | None,
    window_end: datetime | None,
    news_blackouts: list[tuple[datetime, datetime]] | None,
    session_weekdays_only: bool = True,
) -> dict[str, Any]:
    strict = env_bool("BACKTEST_PREFLIGHT_STRICT", True)
    require_seq = env_bool("BACKTEST_PREFLIGHT_REQUIRE_SEQ", True)
    min_quote_coverage = max(0.0, min(1.0, env_float("BACKTEST_PREFLIGHT_MIN_QUOTE_COVERAGE", 0.90)))
    min_depth_coverage = max(0.0, min(1.0, env_float("BACKTEST_PREFLIGHT_MIN_DEPTH_COVERAGE", 0.90)))
    min_rows = max(1, env_int("BACKTEST_PREFLIGHT_MIN_ROWS", 1))
    min_ny_rows = max(0, env_int("BACKTEST_PREFLIGHT_MIN_SESSION_ROWS", 1))
    rows = int(scan.rows)
    headers = list(scan.columns)
    errors: list[str] = []

    if not _has_any_column(headers, ("timestamp", "datetime", "time", "date", "ts_event", "ts_recv")):
        errors.append("missing timestamp column")
    if require_seq and not _has_any_column(headers, ("seq", "sequence", "event_seq", "eventSequence")):
        errors.append("missing seq column (required for deterministic replay)")
    if rows < min_rows:
        errors.append(f"rows={rows} below minimum {min_rows}")
    if rows <= 0:
        errors.append("no rows after session/time filtering")

    if scan.parseable_timestamps != rows:
        errors.append(f"unparseable timestamps={rows - scan.parseable_timestamps}")
    if scan.explicit_tz_timestamps != rows:
        errors.append(f"timestamps without explicit timezone={rows - scan.explicit_tz_timestamps}")

    total = max(1, rows)
    quote_cov = float(scan.quote_rows) / total
    depth_cov = float(scan.depth_rows) / total
    session_rows = rows if session_weekdays_only else max(rows, min_ny_rows)
    if quote_cov < min_quote_coverage:
        errors.append(f"quote coverage={quote_cov:.3f} below minimum {min_quote_coverage:.3f}")
    if depth_cov < min_depth_coverage:
        errors.append(f"depth coverage={depth_cov:.3f} below minimum {min_depth_coverage:.3f}")
    if session_rows < min_ny_rows:
        errors.append(f"NY session rows={session_rows} below minimum {min_ny_rows}")

    if env_bool("STRAT_AVOID_NEWS", False):
        news_path = _resolve_backtest_news_path()
        if news_path == "":
            errors.append("STRAT_AVOID_NEWS=true but BACKTEST_NEWS_PATH is empty")
        elif (news_blackouts or []) == []:
            errors.append("news dataset loaded but no blackout intervals overlap the selected window")

    report = {
        "rows": rows,
        "headers": headers,
        "quote_coverage": round(quote_cov, 6),
        "depth_coverage": round(depth_cov, 6),
        "session_rows": session_rows,
        "window_start": window_start.isoformat().replace("+00:00", "Z") if window_start is not None else None,
        "window_end": window_end.isoformat().replace("+00:00", "Z") if window_end is not None else None,
        "news_blackouts": len(news_blackouts or []),
        "strict": strict,
        "source": data_path,
    }
    if strict and errors:
        raise RuntimeError("backtest-preflight-failed: " + "; ".join(errors))
    if errors:
        print("backtest_preflight_warning=" + "; ".join(errors))
    print(
        "backtest_preflight "
        f"rows={report['rows']} quote_cov={report['quote_coverage']:.3f} "
        f"depth_cov={report['depth_coverage']:.3f} session_rows={report['session_rows']} "
        f"news_blackouts={report['news_blackouts']}"
    )
    return report


def _pick_optional(row: dict[str, Any], *keys: str) -> str | None:
    lowered = {str(k).strip().lower(): v for k, v in row.items()}
    for key in keys:
        value = lowered.get(key.lower())
        if value is None:
            continue
        text = str(value).strip()
        if text == "":
            continue
        return text
    return None


def _num_from_payload(payload: dict[str, Any] | None, *keys: str) -> float | None:
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


def _is_major_news_row(row: dict[str, Any]) -> bool:
    raw = _pick_optional(row, "impact", "importance", "severity", "priority", "rank", "impact_level")
    if raw is None:
        return True
    text = raw.strip().lower()
    if any(token in text for token in ("high", "major", "red", "critical")):
        return True
    try:
        return float(text) >= 3.0
    except ValueError:
        return False


def _merge_intervals(intervals: list[tuple[datetime, datetime]]) -> list[tuple[datetime, datetime]]:
    if not intervals:
        return []
    normalized: list[tuple[datetime, datetime]] = []
    for start, end in intervals:
        a = start.astimezone(timezone.utc)
        b = end.astimezone(timezone.utc)
        if b < a:
            a, b = b, a
        normalized.append((a, b))
    normalized.sort(key=lambda x: x[0])
    merged: list[tuple[datetime, datetime]] = []
    for start, end in normalized:
        if not merged:
            merged.append((start, end))
            continue
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
            continue
        merged.append((start, end))
    return merged


def _resolve_backtest_news_path() -> str:
    return (os.getenv("BACKTEST_NEWS_PATH") or "").strip()


def _iter_news_rows(path: Path) -> list[dict[str, Any]]:
    if (not path.is_dir()) and path.suffix.lower() != ".parquet":
        raise RuntimeError(
            f"Unsupported news data path: {path}. BACKTEST_NEWS_PATH must point to parquet data."
        )
    try:
        import pyarrow.dataset as ds  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - runtime dependency path
        raise RuntimeError(
            "Parquet news loading requires `pyarrow`. Install with `py -3.11 -m pip install pyarrow`."
        ) from exc
    table = ds.dataset(str(path), format="parquet").to_table()
    cols = table.to_pydict()
    if not cols:
        return []
    names = list(cols.keys())
    rows = len(cols[names[0]])
    out: list[dict[str, Any]] = []
    for i in range(rows):
        out.append({name: cols[name][i] for name in names})
    return out


def _load_major_news_blackouts(
    news_path: str | None,
    *,
    pre_minutes: int,
    post_minutes: int,
    window_start: datetime | None,
    window_end: datetime | None,
    major_only: bool = True,
    currencies: tuple[str, ...] = ("USD",),
) -> list[tuple[datetime, datetime]]:
    if news_path is None or news_path.strip() == "":
        return []
    path = Path(news_path.strip())
    if not path.exists():
        raise FileNotFoundError(f"News data not found: {news_path}")

    pre_delta = timedelta(minutes=max(0, int(pre_minutes)))
    post_delta = timedelta(minutes=max(0, int(post_minutes)))
    allow_ccy = {x.strip().upper() for x in currencies if x.strip() != ""}
    intervals: list[tuple[datetime, datetime]] = []
    for row in _iter_news_rows(path):
        if major_only and not _is_major_news_row(row):
            continue
        if allow_ccy:
            ccy = _pick_optional(row, "currency", "ccy", "curr", "country")
            if ccy is not None and ccy.strip().upper() not in allow_ccy:
                continue
        ts_raw = _pick_optional(row, "timestamp", "datetime", "time", "date")
        if ts_raw is None:
            continue
        event_dt = _parse_ts_utc(ts_raw)
        if event_dt is None:
            continue
        start = event_dt - pre_delta
        end = event_dt + post_delta
        if window_start is not None and end < window_start:
            continue
        if window_end is not None and start > window_end:
            continue
        intervals.append((start, end))
    return _merge_intervals(intervals)


def _load_pyarrow_backtest_writers() -> tuple[Any, Any]:
    try:
        import pyarrow as pa  # type: ignore[import-not-found]
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - runtime dependency path
        raise RuntimeError(
            "Backtest parquet output requires `pyarrow`. Install with `py -3.11 -m pip install pyarrow`."
        ) from exc
    return pa, pq


def _parquet_dataset_dir(path: str) -> Path:
    raw = Path(path)
    if raw.suffix.lower() == ".parquet":
        return raw.with_suffix("")
    return raw


def _coerce_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class _ParquetDatasetAppender:
    def __init__(
        self,
        dataset_path: str,
        *,
        columns: list[str],
        int_columns: set[str],
        float_columns: set[str],
        bool_columns: set[str],
        flush_rows: int = 2_000,
    ) -> None:
        self._pa, self._pq = _load_pyarrow_backtest_writers()
        self._columns = [str(col).strip() for col in columns]
        self._int_columns = set(int_columns)
        self._float_columns = set(float_columns)
        self._bool_columns = set(bool_columns)
        self._flush_rows = max(1, int(flush_rows))
        self._rows: list[dict[str, Any]] = []
        self._dataset_dir = _parquet_dataset_dir(dataset_path)
        if self._dataset_dir.exists() and self._dataset_dir.is_file():
            raise RuntimeError(f"Backtest parquet output path must be a directory: {self._dataset_dir}")
        self._dataset_dir.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> str:
        return str(self._dataset_dir)

    def _coerce_value(self, key: str, value: Any) -> Any:
        if key in self._bool_columns:
            return _coerce_bool(value)
        if key in self._int_columns:
            return _coerce_int(value)
        if key in self._float_columns:
            return _coerce_float(value)
        if value is None:
            return None
        return str(value)

    def append_row(self, row: dict[str, Any]) -> None:
        self._rows.append({key: self._coerce_value(key, row.get(key)) for key in self._columns})
        if len(self._rows) >= self._flush_rows:
            self.flush()

    def append_rows(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        for row in rows:
            self.append_row(row)

    def flush(self) -> None:
        if not self._rows:
            return
        arrays = []
        fields = []
        for key in self._columns:
            if key in self._bool_columns:
                dtype = self._pa.bool_()
            elif key in self._int_columns:
                dtype = self._pa.int64()
            elif key in self._float_columns:
                dtype = self._pa.float64()
            else:
                dtype = self._pa.string()
            fields.append(self._pa.field(key, dtype))
            arrays.append(self._pa.array([row.get(key) for row in self._rows], type=dtype))
        schema = self._pa.schema(fields)
        table = self._pa.Table.from_arrays(arrays, schema=schema)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        part = self._dataset_dir / f"part-{stamp}-{uuid.uuid4().hex}.parquet"
        self._pq.write_table(table, str(part), compression="zstd")
        self._rows = []

    def close(self) -> None:
        self.flush()


_CANDIDATE_BOOL_COLUMNS = {
    "has_recent_sweep",
    "bias_ok",
    "continuation",
    "reversal",
    "equal_levels",
    "fib_retracement",
    "key_area_proximity",
    "ml_shadow_approved",
}
_CANDIDATE_INT_COLUMNS = {"window_months", "bar_index", "setup_index", "confluence_score"}
_CANDIDATE_FLOAT_COLUMNS = {
    "setup_close",
    "of_imbalance",
    "of_top_bid_size",
    "of_top_ask_size",
    "of_best_bid",
    "of_best_ask",
    "of_spread",
    "of_trade_size",
    "of_trade_price",
    "ml_shadow_score",
}


_MATRIX_BOOL_COLUMNS = {
    "has_recent_sweep",
    "bias_ok",
    "continuation",
    "reversal",
    "equal_levels",
    "fib_retracement",
    "key_area_proximity",
    "ml_shadow_approved",
    "win",
    "loss",
}
_MATRIX_INT_COLUMNS = {
    "window_months",
    "setup_index",
    "entry_bar_index",
    "exit_bar_index",
    "entry_count",
    "exit_count",
    "size",
    "sl_ticks_abs",
    "tp_ticks_abs",
    "confluence_score",
}
_MATRIX_FLOAT_COLUMNS = {
    "setup_close",
    "of_imbalance",
    "of_top_bid_size",
    "of_top_ask_size",
    "of_best_bid",
    "of_best_ask",
    "of_spread",
    "of_trade_size",
    "of_trade_price",
    "ml_shadow_score",
    "entry_price",
    "exit_price",
    "risk_dollars",
    "gross_entry_size",
    "gross_exit_size",
    "pnl",
    "realized_rr",
}


_SUMMARY_BOOL_COLUMNS = {"orderflow_replay"}
_SUMMARY_INT_COLUMNS = {"window_months", "bars", "num_trades", "news_blackouts"}
_SUMMARY_FLOAT_COLUMNS = {
    "final_equity",
    "net_pnl",
    "return_pct",
    "win_rate_pct",
    "max_drawdown_pct",
}


class _BacktestCandidateParquetWriter:
    def __init__(
        self,
        parquet_path: str,
        *,
        run_id: str,
        strategy: str,
        scenario_id: str,
        window_id: str,
        window_months: int,
        window_start_utc: datetime | None,
        window_end_utc: datetime | None,
    ) -> None:
        self._writer = _ParquetDatasetAppender(
            parquet_path,
            columns=_BACKTEST_CANDIDATE_COLUMNS,
            int_columns=_CANDIDATE_INT_COLUMNS,
            float_columns=_CANDIDATE_FLOAT_COLUMNS,
            bool_columns=_CANDIDATE_BOOL_COLUMNS,
            flush_rows=2_000,
        )
        self._run_id = run_id
        self._strategy = strategy
        self._scenario_id = scenario_id
        self._window_id = window_id
        self._window_months = int(window_months)
        self._window_start_utc = window_start_utc
        self._window_end_utc = window_end_utc

    @property
    def path(self) -> str:
        return self._writer.path

    def append(self, event: dict[str, Any]) -> None:
        row: dict[str, Any] = {
            "run_id": self._run_id,
            "strategy": self._strategy,
            "scenario_id": self._scenario_id,
            "window_id": self._window_id,
            "window_start_utc": (
                self._window_start_utc.isoformat().replace("+00:00", "Z")
                if self._window_start_utc is not None
                else None
            ),
            "window_end_utc": (
                self._window_end_utc.isoformat().replace("+00:00", "Z")
                if self._window_end_utc is not None
                else None
            ),
            "window_months": self._window_months,
            "event_name": event.get("event_name"),
            "bar_index": event.get("bar_index"),
            "bar_ts": event.get("bar_ts"),
            "candidate_id": event.get("candidate_id"),
            "status": event.get("status"),
            "reason": event.get("reason"),
            "side": event.get("side"),
            "setup_index": event.get("setup_index"),
            "setup_ts": event.get("setup_ts"),
            "setup_close": event.get("setup_close"),
            "has_recent_sweep": event.get("has_recent_sweep"),
            "htf_bias": event.get("htf_bias"),
            "bias_ok": event.get("bias_ok"),
            "continuation": event.get("continuation"),
            "reversal": event.get("reversal"),
            "equal_levels": event.get("equal_levels"),
            "fib_retracement": event.get("fib_retracement"),
            "key_area_proximity": event.get("key_area_proximity"),
            "confluence_score": event.get("confluence_score"),
            "of_imbalance": event.get("of_imbalance"),
            "of_top_bid_size": event.get("of_top_bid_size"),
            "of_top_ask_size": event.get("of_top_ask_size"),
            "of_best_bid": event.get("of_best_bid"),
            "of_best_ask": event.get("of_best_ask"),
            "of_spread": event.get("of_spread"),
            "of_trade_size": event.get("of_trade_size"),
            "of_trade_price": event.get("of_trade_price"),
            "ml_shadow_score": event.get("ml_shadow_score"),
            "ml_shadow_reason": event.get("ml_shadow_reason"),
            "ml_shadow_approved": event.get("ml_shadow_approved"),
            "source": event.get("source"),
        }
        self._writer.append_row(row)

    def close(self) -> None:
        self._writer.close()


class _BacktestMatrixParquetWriter:
    def __init__(self, parquet_path: str) -> None:
        self._writer = _ParquetDatasetAppender(
            parquet_path,
            columns=_BACKTEST_MATRIX_COLUMNS,
            int_columns=_MATRIX_INT_COLUMNS,
            float_columns=_MATRIX_FLOAT_COLUMNS,
            bool_columns=_MATRIX_BOOL_COLUMNS,
            flush_rows=250,
        )

    @property
    def path(self) -> str:
        return self._writer.path

    def append_rows(self, rows: list[dict[str, Any]]) -> None:
        self._writer.append_rows(rows)
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()


class _BacktestSummaryParquetWriter:
    def __init__(self, parquet_path: str) -> None:
        self._writer = _ParquetDatasetAppender(
            parquet_path,
            columns=_BACKTEST_SUMMARY_COLUMNS,
            int_columns=_SUMMARY_INT_COLUMNS,
            float_columns=_SUMMARY_FLOAT_COLUMNS,
            bool_columns=_SUMMARY_BOOL_COLUMNS,
            flush_rows=50,
        )

    @property
    def path(self) -> str:
        return self._writer.path

    def append(self, payload: dict[str, Any]) -> None:
        row = {col: payload.get(col) for col in _BACKTEST_SUMMARY_COLUMNS}
        self._writer.append_row(row)
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()


class _BacktestMatrixTracker:
    def __init__(
        self,
        *,
        run_id: str,
        strategy: str,
        scenario_id: str,
        window_id: str,
        window_months: int,
        window_start_utc: datetime | None,
        window_end_utc: datetime | None,
        tick_value: float,
    ) -> None:
        self._run_id = run_id
        self._strategy = strategy
        self._scenario_id = scenario_id
        self._window_id = window_id
        self._window_months = int(window_months)
        self._window_start_utc = window_start_utc
        self._window_end_utc = window_end_utc
        self._tick_value = float(tick_value)
        self._rows: dict[str, dict[str, Any]] = {}
        self._order_to_candidate: dict[str, str] = {}
        self._position_to_candidate: dict[str, str] = {}

    def _ensure_row(self, candidate_id: str) -> dict[str, Any]:
        row = self._rows.get(candidate_id)
        if row is not None:
            return row
        row = {
            "run_id": self._run_id,
            "strategy": self._strategy,
            "scenario_id": self._scenario_id,
            "window_id": self._window_id,
            "window_start_utc": (
                self._window_start_utc.isoformat().replace("+00:00", "Z")
                if self._window_start_utc is not None
                else None
            ),
            "window_end_utc": (
                self._window_end_utc.isoformat().replace("+00:00", "Z")
                if self._window_end_utc is not None
                else None
            ),
            "window_months": self._window_months,
            "candidate_id": candidate_id,
            "side": None,
            "setup_index": None,
            "setup_ts": None,
            "setup_close": None,
            "has_recent_sweep": None,
            "htf_bias": None,
            "bias_ok": None,
            "continuation": None,
            "reversal": None,
            "equal_levels": None,
            "fib_retracement": None,
            "key_area_proximity": None,
            "confluence_score": None,
            "of_imbalance": None,
            "of_top_bid_size": None,
            "of_top_ask_size": None,
            "of_best_bid": None,
            "of_best_ask": None,
            "of_spread": None,
            "of_trade_size": None,
            "of_trade_price": None,
            "ml_shadow_score": None,
            "ml_shadow_reason": None,
            "ml_shadow_approved": None,
            "status_final": None,
            "reason_final": None,
            "entry_ts": None,
            "entry_price": None,
            "entry_event": None,
            "entry_reason": None,
            "entry_bar_index": None,
            "entry_order_id": None,
            "entry_position_id": None,
            "size": None,
            "sl_ticks_abs": None,
            "tp_ticks_abs": None,
            "risk_dollars": None,
            "exit_ts": None,
            "exit_price": None,
            "exit_event": None,
            "exit_reason": None,
            "exit_bar_index": None,
            "exit_order_id": None,
            "exit_position_id": None,
            "entry_count": 0,
            "exit_count": 0,
            "gross_entry_size": 0,
            "gross_exit_size": 0,
            "pnl": None,
            "win": 0,
            "loss": 0,
            "realized_rr": None,
            "result_label": None,
        }
        self._rows[candidate_id] = row
        return row

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _resolve_candidate_id(self, event: dict[str, Any]) -> str | None:
        candidate_id = event.get("candidate_id")
        if candidate_id is not None and str(candidate_id).strip() != "":
            return str(candidate_id)
        order_id = event.get("order_id")
        if order_id is not None and str(order_id).strip() != "":
            mapped = self._order_to_candidate.get(str(order_id))
            if mapped:
                return mapped
        position_id = event.get("position_id")
        if position_id is not None and str(position_id).strip() != "":
            mapped = self._position_to_candidate.get(str(position_id))
            if mapped:
                return mapped
        return None

    def on_candidate_event(self, event: dict[str, Any]) -> None:
        candidate_id = event.get("candidate_id")
        if candidate_id is None or str(candidate_id).strip() == "":
            return
        cid = str(candidate_id)
        row = self._ensure_row(cid)
        row["side"] = event.get("side", row["side"])
        row["setup_index"] = event.get("setup_index", row["setup_index"])
        row["setup_ts"] = event.get("setup_ts", row["setup_ts"])
        row["setup_close"] = event.get("setup_close", row["setup_close"])
        row["has_recent_sweep"] = event.get("has_recent_sweep", row["has_recent_sweep"])
        row["htf_bias"] = event.get("htf_bias", row["htf_bias"])
        row["bias_ok"] = event.get("bias_ok", row["bias_ok"])
        row["continuation"] = event.get("continuation", row["continuation"])
        row["reversal"] = event.get("reversal", row["reversal"])
        row["equal_levels"] = event.get("equal_levels", row["equal_levels"])
        row["fib_retracement"] = event.get("fib_retracement", row["fib_retracement"])
        row["key_area_proximity"] = event.get("key_area_proximity", row["key_area_proximity"])
        row["confluence_score"] = event.get("confluence_score", row["confluence_score"])
        row["of_imbalance"] = event.get("of_imbalance", row["of_imbalance"])
        row["of_top_bid_size"] = event.get("of_top_bid_size", row["of_top_bid_size"])
        row["of_top_ask_size"] = event.get("of_top_ask_size", row["of_top_ask_size"])
        row["of_best_bid"] = event.get("of_best_bid", row["of_best_bid"])
        row["of_best_ask"] = event.get("of_best_ask", row["of_best_ask"])
        row["of_spread"] = event.get("of_spread", row["of_spread"])
        row["of_trade_size"] = event.get("of_trade_size", row["of_trade_size"])
        row["of_trade_price"] = event.get("of_trade_price", row["of_trade_price"])
        row["ml_shadow_score"] = event.get("ml_shadow_score", row["ml_shadow_score"])
        row["ml_shadow_reason"] = event.get("ml_shadow_reason", row["ml_shadow_reason"])
        row["ml_shadow_approved"] = event.get("ml_shadow_approved", row["ml_shadow_approved"])
        row["status_final"] = event.get("status", row["status_final"])
        row["reason_final"] = event.get("reason", row["reason_final"])

    def on_execution_event(self, event: dict[str, Any]) -> None:
        event_name = str(event.get("event_name") or "").strip().lower()
        candidate_id = self._resolve_candidate_id(event)
        if candidate_id is None:
            return
        row = self._ensure_row(candidate_id)
        order_id = event.get("order_id")
        position_id = event.get("position_id")
        if order_id is not None and str(order_id).strip() != "":
            self._order_to_candidate[str(order_id)] = candidate_id
        if position_id is not None and str(position_id).strip() != "":
            self._position_to_candidate[str(position_id)] = candidate_id

        if event_name in {"enter", "tick_enter"}:
            size = max(0, self._safe_int(event.get("size")) or 0)
            row["entry_count"] = int(row.get("entry_count") or 0) + 1
            row["gross_entry_size"] = int(row.get("gross_entry_size") or 0) + size
            row["entry_ts"] = row["entry_ts"] or event.get("bar_ts")
            row["entry_price"] = self._safe_float(event.get("entry_price"))
            row["entry_event"] = event.get("event_name", row["entry_event"])
            row["entry_reason"] = event.get("reason", row["entry_reason"])
            row["entry_bar_index"] = event.get("bar_index", row["entry_bar_index"])
            row["entry_order_id"] = event.get("order_id", row["entry_order_id"])
            row["entry_position_id"] = event.get("position_id", row["entry_position_id"])
            row["size"] = int(row.get("gross_entry_size") or 0)
            row["sl_ticks_abs"] = self._safe_int(event.get("sl_ticks_abs")) or row.get("sl_ticks_abs")
            row["tp_ticks_abs"] = self._safe_int(event.get("tp_ticks_abs")) or row.get("tp_ticks_abs")
            if row["size"] is not None and row["sl_ticks_abs"] is not None and row["sl_ticks_abs"] > 0:
                row["risk_dollars"] = round(float(row["size"]) * float(row["sl_ticks_abs"]) * self._tick_value, 6)
            return

        if event_name in {
            "exit",
            "tick_exit",
            "protective_exit",
            "protective_exit_tick",
            "halt_exit",
            "halt_exit_tick",
            "force_close_end_of_data",
        }:
            size = max(0, self._safe_int(event.get("size")) or 0)
            row["exit_count"] = int(row.get("exit_count") or 0) + 1
            row["gross_exit_size"] = int(row.get("gross_exit_size") or 0) + size
            row["exit_ts"] = event.get("bar_ts", row["exit_ts"])
            row["exit_price"] = self._safe_float(event.get("exit_price"))
            row["exit_event"] = event.get("event_name", row["exit_event"])
            row["exit_reason"] = event.get("reason", row["exit_reason"])
            row["exit_bar_index"] = event.get("bar_index", row["exit_bar_index"])
            row["exit_order_id"] = event.get("order_id", row["exit_order_id"])
            row["exit_position_id"] = event.get("position_id", row["exit_position_id"])
            pnl = self._safe_float(event.get("pnl"))
            prev_pnl = self._safe_float(row.get("pnl")) or 0.0
            if pnl is not None:
                row["pnl"] = round(prev_pnl + pnl, 6)
            net_pnl = self._safe_float(row.get("pnl"))
            if net_pnl is not None:
                row["win"] = 1 if net_pnl > 0 else 0
                row["loss"] = 1 if net_pnl < 0 else 0
            risk = self._safe_float(row.get("risk_dollars"))
            if net_pnl is not None and risk is not None and risk > 0:
                row["realized_rr"] = round(net_pnl / risk, 6)
            if net_pnl is None:
                row["result_label"] = row.get("result_label") or "trade-no-pnl"
            elif net_pnl > 0:
                row["result_label"] = "win"
            elif net_pnl < 0:
                row["result_label"] = "loss"
            else:
                row["result_label"] = "flat"

    def finalize_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in self._rows.values():
            if row["size"] is None:
                row["size"] = int(row.get("gross_entry_size") or 0)
            if row["result_label"] is None:
                if row["entry_count"] == 0:
                    row["result_label"] = "no-trade"
                elif row["exit_count"] == 0:
                    row["result_label"] = "open"
                else:
                    row["result_label"] = "unknown"
            rows.append(row)
        rows.sort(key=lambda r: (str(r.get("setup_ts") or ""), str(r.get("candidate_id") or "")))
        return rows


def _strategy_side_name(side: int | None) -> str:
    if side == 0:
        return "buy"
    if side == 1:
        return "sell"
    return "unknown"


@dataclass(frozen=True)
class _CandidateSetupAdapter:
    side: int
    has_recent_sweep: bool
    bias_ok: bool
    continuation: bool
    reversal: bool
    equal_levels: bool
    fib_retracement: bool
    key_area_proximity: bool
    confluence_score: int


def _event_side_to_int(value: Any) -> int:
    text = str(value or "").strip().lower()
    if text in {"buy", "long", "0"}:
        return BUY
    return SELL


def _event_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _candidate_from_event(event: dict[str, Any]) -> _CandidateSetupAdapter:
    return _CandidateSetupAdapter(
        side=_event_side_to_int(event.get("side")),
        has_recent_sweep=_as_bool(event.get("has_recent_sweep")),
        bias_ok=_as_bool(event.get("bias_ok")),
        continuation=_as_bool(event.get("continuation")),
        reversal=_as_bool(event.get("reversal")),
        equal_levels=_as_bool(event.get("equal_levels")),
        fib_retracement=_as_bool(event.get("fib_retracement")),
        key_area_proximity=_as_bool(event.get("key_area_proximity")),
        confluence_score=_event_int(event.get("confluence_score"), 0),
    )


def _build_backtest_shadow_gate() -> SetupMLGate | None:
    if not env_bool("BACKTEST_SHADOW_ML_ENABLED", False):
        return None
    return SetupMLGate(
        enabled=True,
        model_path=(os.getenv("BACKTEST_SHADOW_ML_MODEL_PATH") or os.getenv("STRAT_ML_MODEL_PATH") or "").strip(),
        min_proba=env_float("BACKTEST_SHADOW_ML_MIN_PROBA", env_float("STRAT_ML_MIN_PROBA", 0.55)),
        fail_open=True,
    )


def _apply_shadow_ml(event: dict[str, Any], gate: SetupMLGate | None) -> dict[str, Any]:
    if gate is None:
        return event
    candidate_id = event.get("candidate_id")
    if candidate_id is None or str(candidate_id).strip() == "":
        return event
    setup = _candidate_from_event(event)
    decision = gate.decision(setup)
    out = dict(event)
    out["ml_shadow_score"] = None if decision.score is None else round(float(decision.score), 6)
    out["ml_shadow_reason"] = decision.reason
    out["ml_shadow_approved"] = bool(decision.approved)
    return out


def _resolve_forward_ml_policy() -> str:
    raw = (os.getenv("STRAT_ML_DECISION_POLICY") or "").strip().lower()
    if raw in {"off", "shadow", "enforce"}:
        return raw
    # Backward-compatible default:
    # - if legacy gate flag is enabled, keep enforce behavior
    # - otherwise run strategy-only decisions
    return "enforce" if env_bool("STRAT_ML_GATE_ENABLED", False) else "off"


def _strategy_from_name(
    name: str,
    hold_bars: int,
    *,
    for_forward: bool,
    backtest_orderflow_sniper: bool = False,
    news_blackouts_utc: list[tuple[datetime, datetime]] | None = None,
):
    key = name.strip().lower()
    if key in {"oneshot", "one_shot", "one-shot"}:
        return OneShotLongStrategy(hold_bars=hold_bars)
    if key in {"ny_structure", "ny_session", "market_structure", "mnq_ny"}:
        if backtest_orderflow_sniper:
            entry_mode = "tick"
            require_orderflow = True
        elif for_forward:
            entry_mode = (os.getenv("STRAT_ENTRY_MODE") or "tick").strip().lower()
            require_orderflow = env_bool("STRAT_REQUIRE_ORDERFLOW", False)
        else:
            # Fallback path for non-orderflow backtests/train mode:
            # completed-bar confirmations with no orderflow gating.
            entry_mode = "bar"
            require_orderflow = False
        symbol = (os.getenv("SYMBOL") or "MNQ").strip().upper()
        profile = get_symbol_profile(symbol)
        sizing_drawdown = env_float("STRAT_ACCOUNT_MAX_DRAWDOWN", env_float("ACCOUNT_MAX_DRAWDOWN", 2_500.0))
        strategy = NYSessionMarketStructureStrategy(
            size=env_int("SIZE", 1),
            session_start=(os.getenv("STRAT_NY_SESSION_START") or "09:30").strip(),
            session_end=(os.getenv("STRAT_NY_SESSION_END") or "16:00").strip(),
            tz_name=(os.getenv("STRAT_TZ_NAME") or "America/New_York").strip(),
            htf_aggregation=env_int("STRAT_HTF_AGGREGATION", 5),
            bias_aggregation=env_int("STRAT_BIAS_AGGREGATION", 60),
            htf_swing_strength_high=env_int("STRAT_HTF_SWING_HIGH", 5),
            htf_swing_strength_low=env_int("STRAT_HTF_SWING_LOW", 5),
            ltf_swing_strength_high=env_int("STRAT_LTF_SWING_HIGH", 3),
            ltf_swing_strength_low=env_int("STRAT_LTF_SWING_LOW", 3),
            wick_sweep_ratio_min=env_float("STRAT_SWEEP_WICK_MIN", 0.5),
            sweep_expiry_bars=env_int("STRAT_SWEEP_EXPIRY_BARS", 40),
            equal_level_tolerance_bps=env_float("STRAT_EQUAL_LEVEL_TOL_BPS", 8.0),
            key_area_tolerance_bps=env_float("STRAT_KEY_AREA_TOL_BPS", 12.0),
            min_confluence_score=env_int("STRAT_MIN_CONFLUENCE", 1),
            require_orderflow=require_orderflow,
            entry_mode=entry_mode,
            avoid_news=env_bool("STRAT_AVOID_NEWS", False),
            news_blackouts_utc=list(news_blackouts_utc or []),
            news_exit_on_event=env_bool("STRAT_NEWS_EXIT_ON_EVENT", False),
            max_hold_bars=hold_bars,
            tick_setup_expiry_bars=env_int("STRAT_TICK_SETUP_EXPIRY_BARS", 3),
            tick_history_size=env_int("STRAT_TICK_HISTORY_SIZE", 120),
            tick_min_imbalance=env_float("STRAT_TICK_MIN_IMBALANCE", 0.12),
            tick_min_trade_size=env_float("STRAT_TICK_MIN_TRADE_SIZE", 1.0),
            tick_spoof_collapse_ratio=env_float("STRAT_TICK_SPOOF_COLLAPSE", 0.35),
            tick_absorption_min_trades=env_int("STRAT_TICK_ABSORPTION_TRADES", 2),
            tick_iceberg_min_reloads=env_int("STRAT_TICK_ICEBERG_RELOADS", 2),
            symbol=symbol,
            tick_size=env_float("STRAT_TICK_SIZE", profile.tick_size),
            tick_value=env_float("STRAT_TICK_VALUE", profile.tick_value),
            account_max_drawdown=sizing_drawdown,
            max_trade_drawdown_fraction=env_float("STRAT_MAX_TRADE_DRAWDOWN_FRACTION", 0.15),
            risk_min_rrr=env_float("STRAT_MIN_RRR", 3.0),
            risk_max_rrr=env_float("STRAT_MAX_RRR", 10.0),
            sl_noise_buffer_ticks=env_int("STRAT_SL_NOISE_BUFFER_TICKS", 2),
            sl_max_ticks=env_int("STRAT_SL_MAX_TICKS", 200),
            tp_front_run_ticks=env_int("STRAT_TP_FRONT_RUN_TICKS", 2),
            dom_liquidity_wall_size=env_float("STRAT_DOM_LIQUIDITY_WALL_SIZE", profile.dom_liquidity_wall_size),
            ml_min_size_fraction=env_float("STRAT_ML_MIN_SIZE_FRACTION", 0.35),
            ml_size_floor_score=env_float("STRAT_ML_SIZE_FLOOR_SCORE", 0.55),
            ml_size_ceiling_score=env_float("STRAT_ML_SIZE_CEILING_SCORE", 0.90),
            enable_exhaustion_market_exit=env_bool("STRAT_ENABLE_EXHAUSTION_MARKET_EXIT", True),
            ml_decision_policy=(
                _resolve_forward_ml_policy()
                if for_forward and (not backtest_orderflow_sniper)
                else "off"
            ),
        )
        if for_forward and (not backtest_orderflow_sniper) and strategy.ml_decision_policy in {"shadow", "enforce"}:
            strategy.set_ml_gate(
                SetupMLGate(
                    enabled=True,
                    model_path=(os.getenv("STRAT_ML_MODEL_PATH") or "").strip(),
                    min_proba=env_float("STRAT_ML_MIN_PROBA", 0.55),
                    fail_open=(
                        True
                        if strategy.ml_decision_policy == "shadow"
                        else env_bool("STRAT_ML_FAIL_OPEN", False)
                    ),
                )
            )
        return strategy
    raise ValueError(f"Unsupported strategy: {name}")


def _resolve_data_path(explicit: str | None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    return must_env("BACKTEST_DATA_PARQUET")


def _parse_float_list(raw: str) -> list[float]:
    out: list[float] = []
    for token in raw.split(","):
        text = token.strip()
        if text == "":
            continue
        try:
            out.append(float(text))
        except ValueError:
            continue
    return out


def _parse_int_list(raw: str) -> list[int]:
    out: list[int] = []
    for token in raw.split(","):
        text = token.strip()
        if text == "":
            continue
        try:
            out.append(int(text))
        except ValueError:
            continue
    return out


def _build_backtest_scenarios(base_cfg: BacktestConfig) -> list[_RunScenario]:
    scenarios: list[_RunScenario] = [_RunScenario(scenario_id="base", config=base_cfg)]
    if not env_bool("BACKTEST_SENSITIVITY_SWEEP", False):
        return scenarios

    delay_values = _parse_int_list(os.getenv("BACKTEST_SWEEP_ENTRY_DELAYS") or "0,1,2")
    slip_values = _parse_float_list(os.getenv("BACKTEST_SWEEP_SLIP_ENTRY_TICKS") or "0,0.5,1.0")
    spread_values = _parse_float_list(os.getenv("BACKTEST_SWEEP_SPREAD_SLIP_K") or "0,1,2")

    for delay in delay_values:
        sid = f"latency_d{delay}"
        scenarios.append(
            _RunScenario(
                scenario_id=sid,
                config=BacktestConfig(
                    initial_cash=base_cfg.initial_cash,
                    fee_per_order=base_cfg.fee_per_order,
                    slippage_bps=base_cfg.slippage_bps,
                    tick_size=base_cfg.tick_size,
                    max_drawdown_abs=base_cfg.max_drawdown_abs,
                    slip_entry_ticks=base_cfg.slip_entry_ticks,
                    slip_stop_ticks=base_cfg.slip_stop_ticks,
                    slip_tp_ticks=base_cfg.slip_tp_ticks,
                    spread_slip_k=base_cfg.spread_slip_k,
                    entry_delay_events=max(0, int(delay)),
                ),
            )
        )

    for slip in slip_values:
        sid = f"slip_e{slip:g}"
        scenarios.append(
            _RunScenario(
                scenario_id=sid,
                config=BacktestConfig(
                    initial_cash=base_cfg.initial_cash,
                    fee_per_order=base_cfg.fee_per_order,
                    slippage_bps=base_cfg.slippage_bps,
                    tick_size=base_cfg.tick_size,
                    max_drawdown_abs=base_cfg.max_drawdown_abs,
                    slip_entry_ticks=float(slip),
                    slip_stop_ticks=float(slip),
                    slip_tp_ticks=float(slip),
                    spread_slip_k=base_cfg.spread_slip_k,
                    entry_delay_events=base_cfg.entry_delay_events,
                ),
            )
        )

    for spread_k in spread_values:
        sid = f"spread_k{spread_k:g}"
        scenarios.append(
            _RunScenario(
                scenario_id=sid,
                config=BacktestConfig(
                    initial_cash=base_cfg.initial_cash,
                    fee_per_order=base_cfg.fee_per_order,
                    slippage_bps=base_cfg.slippage_bps,
                    tick_size=base_cfg.tick_size,
                    max_drawdown_abs=base_cfg.max_drawdown_abs,
                    slip_entry_ticks=base_cfg.slip_entry_ticks,
                    slip_stop_ticks=base_cfg.slip_stop_ticks,
                    slip_tp_ticks=base_cfg.slip_tp_ticks,
                    spread_slip_k=float(spread_k),
                    entry_delay_events=base_cfg.entry_delay_events,
                ),
            )
        )

    deduped: list[_RunScenario] = []
    seen: set[str] = set()
    for scenario in scenarios:
        if scenario.scenario_id in seen:
            continue
        seen.add(scenario.scenario_id)
        deduped.append(scenario)
    return deduped


def _assess_backtest_health(
    *,
    strategy_name: str,
    rows: int,
    candidates: int,
    entered_candidates: int,
    matrix_rows: int,
    trades: int,
    drawdown_pct: float,
) -> tuple[str, list[str]]:
    strategy_key = strategy_name.strip().lower()
    ny_like = strategy_key in {"ny_structure", "ny_session", "market_structure", "mnq_ny"}
    min_candidates_default = 1 if ny_like else 0
    min_trades_default = 1 if ny_like else 0
    min_candidates = max(0, env_int("BACKTEST_HEALTH_MIN_CANDIDATES", min_candidates_default))
    min_trades = max(0, env_int("BACKTEST_HEALTH_MIN_TRADES", min_trades_default))
    max_drawdown_pct = max(0.0, env_float("BACKTEST_HEALTH_MAX_DRAWDOWN_PCT", 50.0))

    reasons: list[str] = []
    if int(rows) <= 0:
        reasons.append("empty-window")
    if int(candidates) < min_candidates:
        reasons.append(f"low-candidates:{int(candidates)}<{min_candidates}")
    if int(trades) < min_trades:
        reasons.append(f"low-trades:{int(trades)}<{min_trades}")
    if int(candidates) > 0 and int(matrix_rows) == 0:
        reasons.append("missing-matrix-rows")
    if max_drawdown_pct > 0 and float(drawdown_pct) > max_drawdown_pct:
        reasons.append(f"drawdown-breach:{float(drawdown_pct):.2f}>{max_drawdown_pct:.2f}")
    if int(entered_candidates) > int(candidates):
        reasons.append("entered-candidates-exceed-candidates")

    return ("ok", []) if not reasons else ("warning", reasons)


def _emit_backtest_health(
    *,
    telemetry: TelemetryRouter,
    strategy_name: str,
    scenario_id: str,
    window_id: str,
    rows: int,
    candidates: int,
    entered_candidates: int,
    matrix_rows: int,
    trades: int,
    net_pnl: float,
    drawdown_pct: float,
    orderflow_replay: bool,
) -> tuple[str, list[str]]:
    status, reasons = _assess_backtest_health(
        strategy_name=strategy_name,
        rows=rows,
        candidates=candidates,
        entered_candidates=entered_candidates,
        matrix_rows=matrix_rows,
        trades=trades,
        drawdown_pct=drawdown_pct,
    )
    reason_text = "|".join(reasons) if reasons else "none"
    telemetry.emit_performance(
        {
            "event_name": "backtest_health",
            "scenario_id": scenario_id,
            "window_id": window_id,
            "status": status,
            "rows": rows,
            "candidates": candidates,
            "entered_candidates": entered_candidates,
            "matrix_rows": matrix_rows,
            "trades": trades,
            "net_pnl": round(float(net_pnl), 6),
            "drawdown_pct": round(float(drawdown_pct), 6),
            "orderflow_replay": bool(orderflow_replay),
            "reasons": reason_text,
        }
    )
    print(
        f"backtest_health status={status} scenario={scenario_id} window_id={window_id} "
        f"rows={rows} candidates={candidates} entered={entered_candidates} matrix_rows={matrix_rows} "
        f"trades={trades} net_pnl={float(net_pnl):.2f} drawdown_pct={float(drawdown_pct):.2f} "
        f"reasons={reason_text}"
    )
    return status, reasons


def run_forward(config: RuntimeConfig, strategy_name: str, hold_bars: int, profile: str) -> None:
    from trading_algo.runtime.bot_runtime import run as run_forward_runtime

    enabled = env_bool("BOT_ENABLED", False)
    environment = (os.getenv("TRADING_ENVIRONMENT") or "DEMO").strip()
    strategy = _strategy_from_name(strategy_name, hold_bars, for_forward=True)
    telemetry = TelemetryRouter.from_env(profile=profile, mode="forward", strategy=strategy_name)
    print(f"TRADING_ENVIRONMENT = {environment}")
    print(f"BROKER             = {config.broker}")
    print(f"RUNTIME_PROFILE    = {profile}")
    print(f"BOT_ENABLED        = {enabled}")
    print(f"SYMBOL             = {config.symbol}")
    print(f"ACCOUNT_ID         = {config.account_id}")
    print(f"LIVE               = {config.live}")
    print(f"STRATEGY           = {strategy_name}")
    print(f"TRADE_ON_START     = {config.trade_on_start}")
    if not enabled:
        print("BOT_ENABLED=0 -> Trading disabled.")
        return
    run_forward_runtime(config, strategy=strategy, profile=profile, telemetry=telemetry)


def run_backtest(data_path: str, strategy_name: str, hold_bars: int, profile: str) -> None:
    raw_backtest_months = env_int("BACKTEST_CANDIDATE_MONTHS", 0)
    backtest_months = int(raw_backtest_months) if int(raw_backtest_months) > 0 else 0
    telemetry = TelemetryRouter.from_env(profile=profile, mode="backtest", strategy=strategy_name)
    default_candidates_parquet = os.path.join(telemetry.cfg.telemetry_dir, "backtest_candidates.parquet")
    candidates_parquet_path = (
        (os.getenv("BACKTEST_CANDIDATES_PARQUET") or default_candidates_parquet).strip() or default_candidates_parquet
    )
    default_matrix_parquet = os.path.join(telemetry.cfg.telemetry_dir, "backtest_candidate_matrix.parquet")
    matrix_parquet_path = (os.getenv("BACKTEST_MATRIX_PARQUET") or default_matrix_parquet).strip() or default_matrix_parquet
    matrix_writer = _BacktestMatrixParquetWriter(matrix_parquet_path)
    default_summary_parquet = os.path.join(telemetry.cfg.telemetry_dir, "backtest_summary.parquet")
    summary_parquet_path = (
        (os.getenv("BACKTEST_SUMMARY_PARQUET") or default_summary_parquet).strip() or default_summary_parquet
    )
    summary_writer = _BacktestSummaryParquetWriter(summary_parquet_path)
    shadow_gate = _build_backtest_shadow_gate()

    symbol = (os.getenv("SYMBOL") or "MNQ").strip().upper()
    symbol_profile = get_symbol_profile(symbol)
    tick_value = env_float("STRAT_TICK_VALUE", symbol_profile.tick_value)
    max_drawdown_abs = env_float("ACCOUNT_MAX_DRAWDOWN_KILLSWITCH", env_float("ACCOUNT_MAX_DRAWDOWN", 0.0))
    base_cfg = BacktestConfig(
        initial_cash=env_float("BACKTEST_INITIAL_CASH", 10_000.0),
        fee_per_order=env_float("BACKTEST_FEE_PER_ORDER", 1.0),
        slippage_bps=env_float("BACKTEST_SLIPPAGE_BPS", 1.0),
        tick_size=env_float("STRAT_TICK_SIZE", symbol_profile.tick_size),
        max_drawdown_abs=max_drawdown_abs if max_drawdown_abs > 0 else None,
        slip_entry_ticks=env_float("BACKTEST_SLIP_ENTRY_TICKS", 0.0),
        slip_stop_ticks=env_float("BACKTEST_SLIP_STOP_TICKS", env_float("BACKTEST_SLIP_ENTRY_TICKS", 0.0)),
        slip_tp_ticks=env_float("BACKTEST_SLIP_TP_TICKS", env_float("BACKTEST_SLIP_ENTRY_TICKS", 0.0)),
        spread_slip_k=env_float("BACKTEST_SPREAD_SLIP_K", 1.0),
        entry_delay_events=max(0, env_int("BACKTEST_ENTRY_DELAY_EVENTS", 1)),
    )
    scenarios = _build_backtest_scenarios(base_cfg)
    replay_bar_sec = max(1, env_int("BACKTEST_BAR_SEC", env_int("STRAT_FORWARD_BAR_SEC", 60)))
    orderflow_session_only = env_bool("BACKTEST_ORDERFLOW_SESSION_ONLY", True)
    orderflow_weekdays_only = bool(orderflow_session_only)
    orderflow_session_start = (os.getenv("STRAT_NY_SESSION_START") or "09:30").strip()
    orderflow_session_end = (os.getenv("STRAT_NY_SESSION_END") or "16:00").strip()
    orderflow_session_tz = (os.getenv("STRAT_TZ_NAME") or "America/New_York").strip() or "America/New_York"

    walk_forward = env_bool("BACKTEST_WALK_FORWARD", False)
    wf_window_default = backtest_months if backtest_months > 0 else 6
    wf_window_months = max(1, env_int("BACKTEST_WF_WINDOW_MONTHS", wf_window_default))
    wf_step_months = max(1, env_int("BACKTEST_WF_STEP_MONTHS", 1))
    wf_start = _parse_utc_boundary(os.getenv("BACKTEST_WF_START_UTC") or os.getenv("BACKTEST_WF_START"), end_of_day=False)
    wf_end = _parse_utc_boundary(os.getenv("BACKTEST_WF_END_UTC") or os.getenv("BACKTEST_WF_END"), end_of_day=True)
    max_rows = max(0, env_int("BACKTEST_MAX_ROWS", 0))
    if max_rows > 0:
        print(f"backtest_row_cap enabled=True rows={max_rows} stage=load")

    total_rows = 0
    total_trades = 0
    total_net_pnl = 0.0
    total_matrix_rows = 0
    run_count = 0
    health_strict = env_bool("BACKTEST_HEALTH_STRICT", False)
    health_warnings = 0
    parquet_batch_size = max(1, env_int("BACKTEST_PARQUET_BATCH_SIZE", 200_000))
    orderflow_parquet_input = is_orderflow_parquet_path(data_path)

    if orderflow_parquet_input:
        if walk_forward:
            raise RuntimeError(
                "BACKTEST_WALK_FORWARD is not supported with streaming Parquet input yet. "
                "Disable walk-forward for this run."
            )

        stream_start_utc: datetime | None = None
        if backtest_months > 0:
            latest_scan = scan_orderflow_parquet(
                data_path,
                session_start=orderflow_session_start if orderflow_session_only else None,
                session_end=orderflow_session_end if orderflow_session_only else None,
                tz_name=orderflow_session_tz,
                weekdays_only=orderflow_weekdays_only,
                max_rows=max_rows if max_rows > 0 else None,
                batch_size=parquet_batch_size,
            )
            if latest_scan.rows <= 0:
                message = f"No rows loaded from orderflow Parquet: {data_path}"
                if orderflow_session_only:
                    raise RuntimeError(
                        f"{message}. Session filter removed all rows "
                        f"(session={orderflow_session_start}-{orderflow_session_end}, tz={orderflow_session_tz}). "
                        "Set BACKTEST_ORDERFLOW_SESSION_ONLY=false to replay all hours, "
                        "or adjust STRAT_NY_SESSION_START/STRAT_NY_SESSION_END/STRAT_TZ_NAME."
                    )
                raise RuntimeError(message)
            if latest_scan.last_ts_utc is None:
                raise RuntimeError(f"No parseable orderflow timestamps loaded from Parquet: {data_path}")
            stream_start_utc = _subtract_months(latest_scan.last_ts_utc, backtest_months)

        parquet_scan = scan_orderflow_parquet(
            data_path,
            session_start=orderflow_session_start if orderflow_session_only else None,
            session_end=orderflow_session_end if orderflow_session_only else None,
            tz_name=orderflow_session_tz,
            weekdays_only=orderflow_weekdays_only,
            max_rows=max_rows if max_rows > 0 else None,
            batch_size=parquet_batch_size,
            start_utc=stream_start_utc,
        )
        if parquet_scan.rows <= 0:
            message = f"No rows loaded from orderflow Parquet: {data_path}"
            if orderflow_session_only:
                raise RuntimeError(
                    f"{message}. Session filter removed all rows "
                    f"(session={orderflow_session_start}-{orderflow_session_end}, tz={orderflow_session_tz}). "
                    "Set BACKTEST_ORDERFLOW_SESSION_ONLY=false to replay all hours, "
                    "or adjust STRAT_NY_SESSION_START/STRAT_NY_SESSION_END/STRAT_TZ_NAME."
                )
            raise RuntimeError(message)
        if parquet_scan.depth_rows <= 0:
            raise RuntimeError(
                "Backtest orderflow replay requires depth-capable rows. "
                f"No usable depth data found in Parquet: {data_path}. "
                "Provide bestBidSize/bestAskSize or depth ladders in the source schema."
            )
        if parquet_scan.first_ts_utc is None or parquet_scan.last_ts_utc is None:
            raise RuntimeError(f"No parseable orderflow timestamps loaded from Parquet: {data_path}")

        window_id = f"latest_{backtest_months}m" if backtest_months > 0 else "full_dataset"
        window_months = backtest_months if backtest_months > 0 else 0
        window = _WindowSpec(
            window_id=window_id,
            start_utc=parquet_scan.first_ts_utc,
            end_utc=parquet_scan.last_ts_utc,
            months=window_months,
        )
        windows = [window]
        print(
            f"backtest_plan strategy={strategy_name} orderflow_replay=True "
            f"input_format=parquet walk_forward={walk_forward} windows={len(windows)} scenarios={len(scenarios)}"
        )

        for scenario in scenarios:
            for window in windows:
                news_blackouts: list[tuple[datetime, datetime]] = []
                if env_bool("STRAT_AVOID_NEWS", False):
                    news_path = _resolve_backtest_news_path()
                    if news_path == "":
                        print("news_blackout_warning=STRAT_AVOID_NEWS=true but BACKTEST_NEWS_PATH is empty; no events filtered")
                    news_blackouts = _load_major_news_blackouts(
                        news_path,
                        pre_minutes=env_int("BACKTEST_NEWS_PRE_MIN", 15),
                        post_minutes=env_int("BACKTEST_NEWS_POST_MIN", 15),
                        window_start=window.start_utc,
                        window_end=window.end_utc,
                        major_only=env_bool("BACKTEST_NEWS_MAJOR_ONLY", True),
                        currencies=tuple(
                            x.strip().upper()
                            for x in (os.getenv("BACKTEST_NEWS_CURRENCIES") or "USD").split(",")
                            if x.strip() != ""
                        ),
                    )
                _validate_orderflow_backtest_preflight_parquet(
                    data_path=data_path,
                    scan=parquet_scan,
                    window_start=window.start_utc,
                    window_end=window.end_utc,
                    news_blackouts=news_blackouts,
                    session_weekdays_only=orderflow_weekdays_only,
                )
                candidate_writer = _BacktestCandidateParquetWriter(
                    candidates_parquet_path,
                    run_id=telemetry.cfg.run_id,
                    strategy=strategy_name,
                    scenario_id=scenario.scenario_id,
                    window_id=window.window_id,
                    window_months=window.months,
                    window_start_utc=window.start_utc,
                    window_end_utc=window.end_utc,
                )
                matrix_tracker = _BacktestMatrixTracker(
                    run_id=telemetry.cfg.run_id,
                    strategy=strategy_name,
                    scenario_id=scenario.scenario_id,
                    window_id=window.window_id,
                    window_months=window.months,
                    window_start_utc=window.start_utc,
                    window_end_utc=window.end_utc,
                    tick_value=tick_value,
                )
                run_diag = {"candidates": 0, "entered": 0}

                def _candidate_sink(event: dict[str, Any]) -> None:
                    enriched = _apply_shadow_ml(event, shadow_gate)
                    enriched["scenario_id"] = scenario.scenario_id
                    enriched["window_id"] = window.window_id
                    run_diag["candidates"] += 1
                    if str(enriched.get("status") or "").strip().lower() == "entered":
                        run_diag["entered"] += 1
                    telemetry.emit_candidate(enriched)
                    candidate_writer.append(enriched)
                    matrix_tracker.on_candidate_event(enriched)

                def _execution_sink(event: dict[str, Any]) -> None:
                    enriched = dict(event)
                    enriched["scenario_id"] = scenario.scenario_id
                    enriched["window_id"] = window.window_id
                    telemetry.emit_execution(enriched)
                    matrix_tracker.on_execution_event(enriched)

                strategy = _strategy_from_name(
                    strategy_name,
                    hold_bars,
                    for_forward=False,
                    backtest_orderflow_sniper=True,
                    news_blackouts_utc=news_blackouts,
                )
                if hasattr(strategy, "set_ml_gate"):
                    getattr(strategy, "set_ml_gate")(None)

                result = run_backtest_orderflow(
                    iter_orderflow_ticks_from_parquet(
                        data_path,
                        session_start=orderflow_session_start if orderflow_session_only else None,
                        session_end=orderflow_session_end if orderflow_session_only else None,
                        tz_name=orderflow_session_tz,
                        weekdays_only=orderflow_weekdays_only,
                        max_rows=max_rows if max_rows > 0 else None,
                        batch_size=parquet_batch_size,
                        start_utc=stream_start_utc,
                    ),
                    strategy,
                    scenario.config,
                    bar_sec=replay_bar_sec,
                    telemetry_callback=lambda event: telemetry.emit_performance(event),
                    execution_callback=_execution_sink,
                    candidate_callback=_candidate_sink,
                )
                processed_rows = parquet_scan.rows
                total_rows += processed_rows
                total_trades += result.num_trades
                total_net_pnl += result.net_pnl
                run_count += 1

                telemetry.emit_performance(
                    {
                        "event_name": "backtest_summary",
                        "scenario_id": scenario.scenario_id,
                        "window_id": window.window_id,
                        "bars": processed_rows,
                        "num_trades": result.num_trades,
                        "final_equity": round(result.final_equity, 6),
                        "net_pnl": round(result.net_pnl, 6),
                        "return_pct": round(result.total_return_pct, 6),
                        "win_rate_pct": round(result.win_rate_pct, 6),
                        "max_drawdown_pct": round(result.max_drawdown_pct, 6),
                        "orderflow_replay": True,
                        "news_blackouts": len(news_blackouts),
                    }
                )
                for trade in result.trades:
                    telemetry.emit_execution(
                        {
                            "event_name": "backtest_trade",
                            "scenario_id": scenario.scenario_id,
                            "window_id": window.window_id,
                            "entry_ts": trade.entry_ts,
                            "exit_ts": trade.exit_ts,
                            "side": _strategy_side_name(trade.side),
                            "size": trade.size,
                            "entry_price": round(trade.entry_price, 6),
                            "exit_price": round(trade.exit_price, 6),
                            "pnl": round(trade.pnl, 6),
                        }
                    )
                matrix_rows = matrix_tracker.finalize_rows()
                matrix_writer.append_rows(matrix_rows)
                total_matrix_rows += len(matrix_rows)
                summary_writer.append(
                    {
                        "run_id": telemetry.cfg.run_id,
                        "strategy": strategy_name,
                        "source_path": data_path,
                        "scenario_id": scenario.scenario_id,
                        "window_id": window.window_id,
                        "window_start_utc": window.start_utc.isoformat().replace("+00:00", "Z"),
                        "window_end_utc": window.end_utc.isoformat().replace("+00:00", "Z"),
                        "window_months": window.months,
                        "bars": processed_rows,
                        "num_trades": result.num_trades,
                        "final_equity": round(result.final_equity, 6),
                        "net_pnl": round(result.net_pnl, 6),
                        "return_pct": round(result.total_return_pct, 6),
                        "win_rate_pct": round(result.win_rate_pct, 6),
                        "max_drawdown_pct": round(result.max_drawdown_pct, 6),
                        "orderflow_replay": True,
                        "news_blackouts": len(news_blackouts),
                    }
                )
                health_status, health_reasons = _emit_backtest_health(
                    telemetry=telemetry,
                    strategy_name=strategy_name,
                    scenario_id=scenario.scenario_id,
                    window_id=window.window_id,
                    rows=processed_rows,
                    candidates=int(run_diag["candidates"]),
                    entered_candidates=int(run_diag["entered"]),
                    matrix_rows=len(matrix_rows),
                    trades=result.num_trades,
                    net_pnl=result.net_pnl,
                    drawdown_pct=result.max_drawdown_pct,
                    orderflow_replay=True,
                )
                if health_status != "ok":
                    health_warnings += 1
                    if health_strict:
                        raise RuntimeError(
                            "backtest-health-failed: "
                            f"scenario={scenario.scenario_id} window_id={window.window_id} "
                            f"reasons={'|'.join(health_reasons)}"
                        )
                print("BACKTEST RESULT")
                print(
                    f"scenario={scenario.scenario_id} window_id={window.window_id} "
                    f"bars={processed_rows} trades={result.num_trades} orderflow_replay=True"
                )
                print(
                    f"window={window.start_utc.isoformat().replace('+00:00', 'Z')}.."
                    f"{window.end_utc.isoformat().replace('+00:00', 'Z')} months={window.months}"
                )
                print(f"bar_sec={replay_bar_sec}")
                print(f"news_blackouts={len(news_blackouts)}")
                candidate_writer.close()
                print(f"candidate_parquet={candidate_writer.path}")
                print(f"matrix_parquet={matrix_writer.path} rows={len(matrix_rows)}")
                print(f"final_equity={result.final_equity:.2f}")
                print(f"net_pnl={result.net_pnl:.2f} return_pct={result.total_return_pct:.2f}")
                print(f"win_rate_pct={result.win_rate_pct:.2f} max_drawdown_pct={result.max_drawdown_pct:.2f}")

    else:
        raise RuntimeError(
            "Backtest now requires parquet input and orderflow replay. "
            "Legacy file-based and OHLCV backtest paths were removed."
        )

    print("BACKTEST AGGREGATE")
    print(f"runs={run_count} bars={total_rows} trades={total_trades} net_pnl={total_net_pnl:.2f}")
    matrix_writer.close()
    summary_writer.close()
    print(f"candidate_parquet={candidates_parquet_path}")
    print(f"matrix_parquet={matrix_parquet_path} rows={total_matrix_rows}")
    print(f"summary_parquet={summary_parquet_path}")
    aggregate_health = "ok" if health_warnings == 0 else "warning"
    print(f"backtest_health_aggregate status={aggregate_health} warnings={health_warnings} strict={health_strict}")


def run_train(data_path: str, model_out: str) -> None:
    train_xgboost_from_parquet(data_path, model_out)


def run_mode(options: ModeOptions) -> None:
    mode = options.mode.strip().lower()
    profile = _resolve_profile(options.profile)
    if mode == "forward":
        run_forward(load_runtime_config(), options.strategy, options.hold_bars, profile)
        return

    data_path = _resolve_data_path(options.data_path)
    if mode == "backtest":
        run_backtest(data_path, options.strategy, options.hold_bars, profile)
        return
    if mode == "train":
        run_train(data_path, options.model_out)
        return
    raise ValueError(f"Unsupported mode: {options.mode}. Use forward, backtest, or train.")
