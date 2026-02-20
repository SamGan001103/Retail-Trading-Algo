from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from trading_algo.core import BUY


class SetupLike(Protocol):
    @property
    def side(self) -> int:
        ...

    @property
    def has_recent_sweep(self) -> bool:
        ...

    @property
    def bias_ok(self) -> bool:
        ...

    @property
    def continuation(self) -> bool:
        ...

    @property
    def reversal(self) -> bool:
        ...

    @property
    def equal_levels(self) -> bool:
        ...

    @property
    def fib_retracement(self) -> bool:
        ...

    @property
    def key_area_proximity(self) -> bool:
        ...

    @property
    def confluence_score(self) -> int:
        ...


@dataclass(frozen=True)
class SetupGateDecision:
    approved: bool
    score: float | None
    reason: str


class SetupMLGate:
    def __init__(
        self,
        *,
        enabled: bool,
        model_path: str,
        min_proba: float = 0.55,
        fail_open: bool = False,
    ) -> None:
        self.enabled = bool(enabled)
        self.model_path = model_path.strip()
        self.min_proba = float(min_proba)
        self.fail_open = bool(fail_open)
        self._xgb = None
        self._booster = None
        self._load_error: str | None = None
        if self.enabled:
            self._load_model()

    def _load_model(self) -> None:
        if not self.model_path:
            self._load_error = "model-path-empty"
            return
        path = Path(self.model_path)
        if not path.exists():
            self._load_error = f"model-not-found:{path}"
            return
        try:
            import xgboost as xgb  # type: ignore
        except Exception as exc:  # pragma: no cover - import error branch
            self._load_error = f"xgboost-import-failed:{exc}"
            return
        try:
            booster = xgb.Booster()
            booster.load_model(str(path))
        except Exception as exc:  # pragma: no cover - invalid model branch
            self._load_error = f"model-load-failed:{exc}"
            return
        self._xgb = xgb
        self._booster = booster

    def evaluate(self, setup: SetupLike) -> tuple[bool, float | None, str]:
        decision = self.decision(setup)
        return decision.approved, decision.score, decision.reason

    def decision(self, setup: SetupLike) -> SetupGateDecision:
        if not self.enabled:
            return SetupGateDecision(approved=True, score=None, reason="ml-disabled")
        if self._xgb is None or self._booster is None:
            return SetupGateDecision(
                approved=self.fail_open,
                score=None,
                reason=f"ml-unavailable:{self._load_error or 'unknown'}",
            )

        features = self._to_features(setup)
        dmat = self._xgb.DMatrix([features])
        score = float(self._booster.predict(dmat)[0])
        approved = score >= self.min_proba
        return SetupGateDecision(
            approved=approved,
            score=score,
            reason="ml-approve" if approved else "ml-reject",
        )

    @staticmethod
    def _to_features(setup: SetupLike) -> list[float]:
        return [
            1.0 if setup.side == BUY else 0.0,
            1.0 if setup.has_recent_sweep else 0.0,
            1.0 if setup.bias_ok else 0.0,
            1.0 if setup.continuation else 0.0,
            1.0 if setup.reversal else 0.0,
            1.0 if setup.equal_levels else 0.0,
            1.0 if setup.fib_retracement else 0.0,
            1.0 if setup.key_area_proximity else 0.0,
            float(setup.confluence_score),
        ]
