from __future__ import annotations

import calendar
import csv
import os
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from trading_algo.backtest import (
    BacktestConfig,
    OrderFlowTick,
    load_bars_from_csv,
    load_orderflow_ticks_from_csv,
    run_backtest as run_backtest_sim,
    run_backtest_orderflow,
)
from trading_algo.config import RuntimeConfig, env_bool, env_float, env_int, get_symbol_profile, load_runtime_config, must_env
from trading_algo.core import BUY, SELL
from trading_algo.ml import SetupMLGate, train_xgboost_from_csv
from trading_algo.strategy import MarketBar, NYSessionMarketStructureStrategy, OneShotLongStrategy
from trading_algo.telemetry import TelemetryRouter


@dataclass(frozen=True)
class ModeOptions:
    mode: str
    data_csv: str | None
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
    "source_csv",
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


def _ensure_orderflow_backtest_dataset(ticks: list[OrderFlowTick], path: str) -> None:
    if any(_tick_has_usable_orderflow(tick) for tick in ticks):
        return
    raise RuntimeError(
        "Backtest orderflow replay requires depth-capable rows. "
        f"No usable depth data found in CSV: {path}. "
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


def _in_session_utc(dt_utc: datetime, *, start: time, end: time, tz_name: str) -> bool:
    try:
        dt_local = dt_utc.astimezone(ZoneInfo(tz_name))
    except ZoneInfoNotFoundError:
        dt_local = dt_utc
    if dt_local.weekday() >= 5:
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


def _validate_orderflow_backtest_preflight(
    *,
    data_csv: str,
    ticks: list[OrderFlowTick],
    window_start: datetime | None,
    window_end: datetime | None,
    news_blackouts: list[tuple[datetime, datetime]] | None,
) -> dict[str, Any]:
    strict = env_bool("BACKTEST_PREFLIGHT_STRICT", True)
    require_seq = env_bool("BACKTEST_PREFLIGHT_REQUIRE_SEQ", True)
    min_quote_coverage = max(0.0, min(1.0, env_float("BACKTEST_PREFLIGHT_MIN_QUOTE_COVERAGE", 0.90)))
    min_depth_coverage = max(0.0, min(1.0, env_float("BACKTEST_PREFLIGHT_MIN_DEPTH_COVERAGE", 0.90)))
    min_rows = max(1, env_int("BACKTEST_PREFLIGHT_MIN_ROWS", 1))
    min_ny_rows = max(0, env_int("BACKTEST_PREFLIGHT_MIN_SESSION_ROWS", 1))
    ny_start = _parse_hhmm_optional(os.getenv("STRAT_NY_SESSION_START") or "", "09:30")
    ny_end = _parse_hhmm_optional(os.getenv("STRAT_NY_SESSION_END") or "", "16:00")
    tz_name = (os.getenv("STRAT_TZ_NAME") or "America/New_York").strip() or "America/New_York"

    with Path(data_csv).open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = list(reader.fieldnames or [])
        ts_values: list[str] = []
        for row in reader:
            ts_raw = _pick_optional(row, "timestamp", "datetime", "time", "date")
            if ts_raw is not None:
                ts_values.append(ts_raw)

    errors: list[str] = []
    if not _has_any_column(headers, ("timestamp", "datetime", "time", "date")):
        errors.append("missing timestamp column")
    if require_seq and not _has_any_column(headers, ("seq", "sequence", "event_seq", "eventSequence")):
        errors.append("missing seq column (required for deterministic replay)")
    if len(ticks) < min_rows:
        errors.append(f"rows={len(ticks)} below minimum {min_rows}")

    explicit_tz_count = 0
    parseable_count = 0
    for raw_ts in ts_values:
        if _timestamp_has_explicit_tz(raw_ts):
            explicit_tz_count += 1
        if _parse_ts_utc(raw_ts) is not None:
            parseable_count += 1
    ts_total = len(ts_values)
    if ts_total == 0:
        errors.append("no timestamp values in CSV rows")
    else:
        if parseable_count != ts_total:
            errors.append(f"unparseable timestamps={ts_total - parseable_count}")
        if explicit_tz_count != ts_total:
            errors.append(f"timestamps without explicit timezone={ts_total - explicit_tz_count}")

    monotonic_ok = True
    prev_key: tuple[datetime, int] | None = None
    for i, tick in enumerate(ticks):
        dt = _parse_ts_utc(tick.ts)
        if dt is None:
            monotonic_ok = False
            break
        key = (dt, int(tick.seq) if int(tick.seq) > 0 else i + 1)
        if prev_key is not None and key < prev_key:
            monotonic_ok = False
            break
        prev_key = key
    if not monotonic_ok:
        errors.append("events are not monotonic by (timestamp, seq)")

    total = max(1, len(ticks))
    quote_rows = 0
    depth_rows = 0
    session_rows = 0
    for tick in ticks:
        bid = _num_from_payload(tick.quote, "bid", "bidprice", "bestbid")
        ask = _num_from_payload(tick.quote, "ask", "askprice", "bestask")
        if bid is None:
            bid = _num_from_payload(tick.depth, "bestbid", "bidprice")
        if ask is None:
            ask = _num_from_payload(tick.depth, "bestask", "askprice")
        if bid is not None and ask is not None:
            quote_rows += 1
        if _tick_has_usable_orderflow(tick):
            depth_rows += 1
        ts = _parse_ts_utc(tick.ts)
        if ts is not None and _in_session_utc(ts, start=ny_start, end=ny_end, tz_name=tz_name):
            session_rows += 1

    quote_cov = quote_rows / total
    depth_cov = depth_rows / total
    if quote_cov < min_quote_coverage:
        errors.append(f"quote coverage={quote_cov:.3f} below minimum {min_quote_coverage:.3f}")
    if depth_cov < min_depth_coverage:
        errors.append(f"depth coverage={depth_cov:.3f} below minimum {min_depth_coverage:.3f}")
    if session_rows < min_ny_rows:
        errors.append(f"NY session rows={session_rows} below minimum {min_ny_rows}")

    if env_bool("STRAT_AVOID_NEWS", False):
        news_csv = (os.getenv("BACKTEST_NEWS_CSV") or "").strip()
        if news_csv == "":
            errors.append("STRAT_AVOID_NEWS=true but BACKTEST_NEWS_CSV is empty")
        elif (news_blackouts or []) == []:
            errors.append("news CSV loaded but no blackout intervals overlap the selected window")

    report = {
        "rows": len(ticks),
        "headers": headers,
        "quote_coverage": round(quote_cov, 6),
        "depth_coverage": round(depth_cov, 6),
        "session_rows": session_rows,
        "window_start": window_start.isoformat().replace("+00:00", "Z") if window_start is not None else None,
        "window_end": window_end.isoformat().replace("+00:00", "Z") if window_end is not None else None,
        "news_blackouts": len(news_blackouts or []),
        "strict": strict,
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


def _pick_optional(row: dict[str, str], *keys: str) -> str | None:
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


def _is_major_news_row(row: dict[str, str]) -> bool:
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


def _load_major_news_blackouts(
    csv_path: str | None,
    *,
    pre_minutes: int,
    post_minutes: int,
    window_start: datetime | None,
    window_end: datetime | None,
    major_only: bool = True,
    currencies: tuple[str, ...] = ("USD",),
) -> list[tuple[datetime, datetime]]:
    if csv_path is None or csv_path.strip() == "":
        return []
    path = Path(csv_path.strip())
    if not path.exists():
        raise FileNotFoundError(f"News CSV not found: {csv_path}")

    pre_delta = timedelta(minutes=max(0, int(pre_minutes)))
    post_delta = timedelta(minutes=max(0, int(post_minutes)))
    allow_ccy = {x.strip().upper() for x in currencies if x.strip() != ""}
    intervals: list[tuple[datetime, datetime]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
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


def _read_csv_header(path: Path) -> list[str] | None:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
    except OSError:
        return None
    if header is None:
        return None
    return [str(col).strip() for col in header]


def _rotate_incompatible_csv(path: Path, *, expected: list[str], actual: list[str] | None) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = 0
    while True:
        tag = f".legacy_{stamp}" if suffix == 0 else f".legacy_{stamp}_{suffix}"
        rotated = path.with_name(f"{path.stem}{tag}{path.suffix}")
        if not rotated.exists():
            break
        suffix += 1
    path.replace(rotated)
    expected_text = ",".join(expected)
    actual_text = ",".join(actual) if actual is not None else "<missing-header>"
    print(
        "backtest_csv_header_rotate "
        f"path={path} moved_to={rotated} "
        f"expected={expected_text} found={actual_text}"
    )


def _ensure_csv_header(path: Path, fieldnames: list[str]) -> None:
    expected = [str(name).strip() for name in fieldnames]
    if path.exists() and path.stat().st_size > 0:
        actual = _read_csv_header(path)
        if actual == expected:
            return
        _rotate_incompatible_csv(path, expected=expected, actual=actual)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


class _BacktestCandidateCsvWriter:
    def __init__(
        self,
        csv_path: str,
        *,
        run_id: str,
        strategy: str,
        scenario_id: str,
        window_id: str,
        window_months: int,
        window_start_utc: datetime | None,
        window_end_utc: datetime | None,
    ) -> None:
        self._path = Path(csv_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._run_id = run_id
        self._strategy = strategy
        self._scenario_id = scenario_id
        self._window_id = window_id
        self._window_months = int(window_months)
        self._window_start_utc = window_start_utc
        self._window_end_utc = window_end_utc
        self._ensure_header()

    @property
    def path(self) -> str:
        return str(self._path)

    def _ensure_header(self) -> None:
        _ensure_csv_header(self._path, _BACKTEST_CANDIDATE_COLUMNS)

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
        with self._path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_BACKTEST_CANDIDATE_COLUMNS)
            writer.writerow(row)


class _BacktestMatrixCsvWriter:
    def __init__(self, csv_path: str) -> None:
        self._path = Path(csv_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_header()

    @property
    def path(self) -> str:
        return str(self._path)

    def _ensure_header(self) -> None:
        _ensure_csv_header(self._path, _BACKTEST_MATRIX_COLUMNS)

    def append_rows(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with self._path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_BACKTEST_MATRIX_COLUMNS)
            for row in rows:
                writer.writerow(row)


class _BacktestSummaryCsvWriter:
    def __init__(self, csv_path: str) -> None:
        self._path = Path(csv_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_header()

    @property
    def path(self) -> str:
        return str(self._path)

    def _ensure_header(self) -> None:
        _ensure_csv_header(self._path, _BACKTEST_SUMMARY_COLUMNS)

    def append(self, payload: dict[str, Any]) -> None:
        row = {col: payload.get(col) for col in _BACKTEST_SUMMARY_COLUMNS}
        with self._path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_BACKTEST_SUMMARY_COLUMNS)
            writer.writerow(row)


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


def _resolve_data_csv(explicit: str | None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    return must_env("BACKTEST_DATA_CSV")


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


def run_backtest(data_csv: str, strategy_name: str, hold_bars: int, profile: str) -> None:
    backtest_months = max(1, env_int("BACKTEST_CANDIDATE_MONTHS", 6))
    telemetry = TelemetryRouter.from_env(profile=profile, mode="backtest", strategy=strategy_name)
    default_candidates_csv = os.path.join(telemetry.cfg.telemetry_dir, "backtest_candidates.csv")
    candidates_csv_path = (os.getenv("BACKTEST_CANDIDATES_CSV") or default_candidates_csv).strip() or default_candidates_csv
    default_matrix_csv = os.path.join(telemetry.cfg.telemetry_dir, "backtest_candidate_matrix.csv")
    matrix_csv_path = (os.getenv("BACKTEST_MATRIX_CSV") or default_matrix_csv).strip() or default_matrix_csv
    matrix_writer = _BacktestMatrixCsvWriter(matrix_csv_path)
    default_summary_csv = os.path.join(telemetry.cfg.telemetry_dir, "backtest_summary.csv")
    summary_csv_path = (os.getenv("BACKTEST_SUMMARY_CSV") or default_summary_csv).strip() or default_summary_csv
    summary_writer = _BacktestSummaryCsvWriter(summary_csv_path)
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
    ny_orderflow_backtest = strategy_name.strip().lower() in {"ny_structure", "ny_session", "market_structure", "mnq_ny"}
    replay_bar_sec = max(1, env_int("BACKTEST_BAR_SEC", env_int("STRAT_FORWARD_BAR_SEC", 60)))

    walk_forward = env_bool("BACKTEST_WALK_FORWARD", False)
    wf_window_months = max(1, env_int("BACKTEST_WF_WINDOW_MONTHS", backtest_months))
    wf_step_months = max(1, env_int("BACKTEST_WF_STEP_MONTHS", 1))
    wf_start = _parse_utc_boundary(os.getenv("BACKTEST_WF_START_UTC") or os.getenv("BACKTEST_WF_START"), end_of_day=False)
    wf_end = _parse_utc_boundary(os.getenv("BACKTEST_WF_END_UTC") or os.getenv("BACKTEST_WF_END"), end_of_day=True)

    total_rows = 0
    total_trades = 0
    total_net_pnl = 0.0
    total_matrix_rows = 0
    run_count = 0

    if ny_orderflow_backtest:
        ticks_all = load_orderflow_ticks_from_csv(data_csv)
        _ensure_orderflow_backtest_dataset(ticks_all, data_csv)
        parsed_ticks = _parsed_ticks_utc(ticks_all)
        if not parsed_ticks:
            raise RuntimeError(f"No parseable orderflow timestamps loaded from CSV: {data_csv}")

        windows: list[_WindowSpec]
        window_ticks: dict[str, list[OrderFlowTick]] = {}
        if walk_forward:
            windows = _build_walk_forward_windows(
                parsed_times_utc=[dt for _, dt in parsed_ticks],
                window_months=wf_window_months,
                step_months=wf_step_months,
                start_utc=wf_start,
                end_utc=wf_end,
            )
            if not windows:
                raise RuntimeError(
                    "Walk-forward produced zero windows. "
                    "Adjust BACKTEST_WF_WINDOW_MONTHS/BACKTEST_WF_STEP_MONTHS or the start/end bounds."
                )
            for window in windows:
                window_ticks[window.window_id] = _ticks_in_window(parsed_ticks, window)
        else:
            selected_ticks, window_start_dt, window_end_dt = _latest_months_window_ticks(ticks_all, backtest_months)
            start_dt = window_start_dt or min(dt for _, dt in parsed_ticks)
            end_dt = window_end_dt or max(dt for _, dt in parsed_ticks)
            single = _WindowSpec(
                window_id=f"latest_{backtest_months}m",
                start_utc=start_dt,
                end_utc=end_dt,
                months=backtest_months,
            )
            windows = [single]
            window_ticks[single.window_id] = selected_ticks
        print(
            f"backtest_plan strategy={strategy_name} orderflow_replay=True "
            f"walk_forward={walk_forward} windows={len(windows)} scenarios={len(scenarios)}"
        )

        for scenario in scenarios:
            for window in windows:
                ticks = window_ticks.get(window.window_id, [])
                if not ticks:
                    continue
                news_blackouts: list[tuple[datetime, datetime]] = []
                if env_bool("STRAT_AVOID_NEWS", False):
                    news_csv = (os.getenv("BACKTEST_NEWS_CSV") or "").strip()
                    if news_csv == "":
                        print("news_blackout_warning=STRAT_AVOID_NEWS=true but BACKTEST_NEWS_CSV is empty; no events filtered")
                    news_blackouts = _load_major_news_blackouts(
                        news_csv,
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
                _validate_orderflow_backtest_preflight(
                    data_csv=data_csv,
                    ticks=ticks,
                    window_start=window.start_utc,
                    window_end=window.end_utc,
                    news_blackouts=news_blackouts,
                )
                csv_writer = _BacktestCandidateCsvWriter(
                    candidates_csv_path,
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

                def _candidate_sink(event: dict[str, Any]) -> None:
                    enriched = _apply_shadow_ml(event, shadow_gate)
                    enriched["scenario_id"] = scenario.scenario_id
                    enriched["window_id"] = window.window_id
                    telemetry.emit_candidate(enriched)
                    csv_writer.append(enriched)
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
                    ticks,
                    strategy,
                    scenario.config,
                    bar_sec=replay_bar_sec,
                    telemetry_callback=lambda event: telemetry.emit_performance(event),
                    execution_callback=_execution_sink,
                    candidate_callback=_candidate_sink,
                )
                processed_rows = len(ticks)
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
                        "source_csv": data_csv,
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
                print(f"candidate_csv={csv_writer.path}")
                print(f"matrix_csv={matrix_writer.path} rows={len(matrix_rows)}")
                print(f"final_equity={result.final_equity:.2f}")
                print(f"net_pnl={result.net_pnl:.2f} return_pct={result.total_return_pct:.2f}")
                print(f"win_rate_pct={result.win_rate_pct:.2f} max_drawdown_pct={result.max_drawdown_pct:.2f}")
    else:
        bars_all = load_bars_from_csv(data_csv)
        parsed_bars = _parsed_bars_utc(bars_all)
        if not parsed_bars:
            raise RuntimeError(f"No parseable bar timestamps loaded from CSV: {data_csv}")

        windows = []
        window_bars: dict[str, list[MarketBar]] = {}
        if walk_forward:
            windows = _build_walk_forward_windows(
                parsed_times_utc=[dt for _, dt in parsed_bars],
                window_months=wf_window_months,
                step_months=wf_step_months,
                start_utc=wf_start,
                end_utc=wf_end,
            )
            if not windows:
                raise RuntimeError(
                    "Walk-forward produced zero windows. "
                    "Adjust BACKTEST_WF_WINDOW_MONTHS/BACKTEST_WF_STEP_MONTHS or the start/end bounds."
                )
            for window in windows:
                window_bars[window.window_id] = _bars_in_window(parsed_bars, window)
        else:
            selected_bars, window_start_dt, window_end_dt = _latest_months_window(bars_all, backtest_months)
            start_dt = window_start_dt or min(dt for _, dt in parsed_bars)
            end_dt = window_end_dt or max(dt for _, dt in parsed_bars)
            single = _WindowSpec(
                window_id=f"latest_{backtest_months}m",
                start_utc=start_dt,
                end_utc=end_dt,
                months=backtest_months,
            )
            windows = [single]
            window_bars[single.window_id] = selected_bars
        print(
            f"backtest_plan strategy={strategy_name} orderflow_replay=False "
            f"walk_forward={walk_forward} windows={len(windows)} scenarios={len(scenarios)}"
        )

        for scenario in scenarios:
            for window in windows:
                bars = window_bars.get(window.window_id, [])
                if not bars:
                    continue
                csv_writer = _BacktestCandidateCsvWriter(
                    candidates_csv_path,
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

                def _candidate_sink(event: dict[str, Any]) -> None:
                    enriched = _apply_shadow_ml(event, shadow_gate)
                    enriched["scenario_id"] = scenario.scenario_id
                    enriched["window_id"] = window.window_id
                    telemetry.emit_candidate(enriched)
                    csv_writer.append(enriched)
                    matrix_tracker.on_candidate_event(enriched)

                def _execution_sink(event: dict[str, Any]) -> None:
                    enriched = dict(event)
                    enriched["scenario_id"] = scenario.scenario_id
                    enriched["window_id"] = window.window_id
                    telemetry.emit_execution(enriched)
                    matrix_tracker.on_execution_event(enriched)

                strategy = _strategy_from_name(strategy_name, hold_bars, for_forward=False)
                result = run_backtest_sim(
                    bars,
                    strategy,
                    scenario.config,
                    telemetry_callback=lambda event: telemetry.emit_performance(event),
                    execution_callback=_execution_sink,
                    candidate_callback=_candidate_sink,
                )
                processed_rows = len(bars)
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
                        "orderflow_replay": False,
                        "news_blackouts": 0,
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
                        "source_csv": data_csv,
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
                        "orderflow_replay": False,
                        "news_blackouts": 0,
                    }
                )
                print("BACKTEST RESULT")
                print(
                    f"scenario={scenario.scenario_id} window_id={window.window_id} "
                    f"bars={processed_rows} trades={result.num_trades} orderflow_replay=False"
                )
                print(
                    f"window={window.start_utc.isoformat().replace('+00:00', 'Z')}.."
                    f"{window.end_utc.isoformat().replace('+00:00', 'Z')} months={window.months}"
                )
                print(f"candidate_csv={csv_writer.path}")
                print(f"matrix_csv={matrix_writer.path} rows={len(matrix_rows)}")
                print(f"final_equity={result.final_equity:.2f}")
                print(f"net_pnl={result.net_pnl:.2f} return_pct={result.total_return_pct:.2f}")
                print(f"win_rate_pct={result.win_rate_pct:.2f} max_drawdown_pct={result.max_drawdown_pct:.2f}")

    print("BACKTEST AGGREGATE")
    print(f"runs={run_count} bars={total_rows} trades={total_trades} net_pnl={total_net_pnl:.2f}")
    print(f"candidate_csv={candidates_csv_path}")
    print(f"matrix_csv={matrix_csv_path} rows={total_matrix_rows}")
    print(f"summary_csv={summary_csv_path}")


def run_train(data_csv: str, model_out: str) -> None:
    train_xgboost_from_csv(data_csv, model_out)


def run_mode(options: ModeOptions) -> None:
    mode = options.mode.strip().lower()
    profile = _resolve_profile(options.profile)
    if mode == "forward":
        run_forward(load_runtime_config(), options.strategy, options.hold_bars, profile)
        return

    data_csv = _resolve_data_csv(options.data_csv)
    if mode == "backtest":
        run_backtest(data_csv, options.strategy, options.hold_bars, profile)
        return
    if mode == "train":
        run_train(data_csv, options.model_out)
        return
    raise ValueError(f"Unsupported mode: {options.mode}. Use forward, backtest, or train.")
