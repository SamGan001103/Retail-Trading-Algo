from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Mapping

from .logging import get_logger


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_int(raw: str | None, default: int) -> int:
    if raw is None:
        return default
    value = raw.strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


@dataclass(frozen=True)
class TelemetryConfig:
    profile: str
    mode: str
    strategy: str
    run_id: str
    telemetry_dir: str
    candidates_file: str
    performance_file: str
    debug_tick_trace_every_n: int


class TelemetryRouter:
    def __init__(self, cfg: TelemetryConfig) -> None:
        self.cfg = cfg
        self._logger = get_logger("trading_algo.telemetry")
        self._debug = cfg.profile.strip().lower() == "debug"
        self._lock = Lock()
        self._candidates_path = os.path.join(cfg.telemetry_dir, cfg.candidates_file)
        self._performance_path = os.path.join(cfg.telemetry_dir, cfg.performance_file)
        os.makedirs(cfg.telemetry_dir, exist_ok=True)

    @classmethod
    def from_env(cls, *, profile: str, mode: str, strategy: str) -> "TelemetryRouter":
        telemetry_dir = (os.getenv("TELEMETRY_DIR") or "artifacts/telemetry").strip()
        candidates_file = (os.getenv("TELEMETRY_CANDIDATES_FILE") or "candidate_trades.jsonl").strip()
        performance_file = (os.getenv("TELEMETRY_PERF_FILE") or "performance.jsonl").strip()
        run_id = (os.getenv("TELEMETRY_RUN_ID") or "").strip() or datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        debug_tick_trace_every_n = max(1, _safe_int(os.getenv("DEBUG_TICK_TRACE_EVERY_N"), 20))
        return cls(
            TelemetryConfig(
                profile=profile.strip().lower(),
                mode=mode.strip().lower(),
                strategy=strategy.strip(),
                run_id=run_id,
                telemetry_dir=telemetry_dir,
                candidates_file=candidates_file,
                performance_file=performance_file,
                debug_tick_trace_every_n=debug_tick_trace_every_n,
            )
        )

    def is_debug(self) -> bool:
        return self._debug

    def debug_tick_trace_every_n(self) -> int:
        return self.cfg.debug_tick_trace_every_n

    def emit_candidate(self, payload: Mapping[str, Any]) -> None:
        self._emit("candidate_trade", payload, self._candidates_path)

    def emit_performance(self, payload: Mapping[str, Any]) -> None:
        self._emit("performance", payload, self._performance_path)

    def emit_execution(self, payload: Mapping[str, Any]) -> None:
        self._emit("execution", payload, self._performance_path)

    def trace(self, message: str, **fields: Any) -> None:
        if not self._debug:
            return
        if fields:
            self._logger.info("%s | %s", message, json.dumps(fields, default=_json_default, sort_keys=True))
            return
        self._logger.info("%s", message)

    def _emit(self, event_type: str, payload: Mapping[str, Any], path: str) -> None:
        record: dict[str, Any] = {
            "ts": _utc_now(),
            "event_type": event_type,
            "run_id": self.cfg.run_id,
            "mode": self.cfg.mode,
            "profile": self.cfg.profile,
            "strategy": self.cfg.strategy,
        }
        record.update(dict(payload))
        line = json.dumps(record, default=_json_default, separators=(",", ":"))
        with self._lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
                f.write("\n")
        if self._debug:
            self._logger.info("%s", line)

