from dataclasses import dataclass

from trading_algo.ml import SetupMLGate
from trading_algo.core import BUY


@dataclass(frozen=True)
class _Setup:
    side: int = BUY
    has_recent_sweep: bool = True
    bias_ok: bool = True
    continuation: bool = True
    reversal: bool = False
    equal_levels: bool = True
    fib_retracement: bool = False
    key_area_proximity: bool = True
    confluence_score: int = 2


def test_setup_ml_gate_disabled_allows():
    gate = SetupMLGate(enabled=False, model_path="")
    approved, score, reason = gate.evaluate(_Setup())
    assert approved is True
    assert score is None
    assert reason == "ml-disabled"


def test_setup_ml_gate_enabled_without_model_fails_closed():
    gate = SetupMLGate(enabled=True, model_path="")
    approved, score, reason = gate.evaluate(_Setup())
    assert approved is False
    assert score is None
    assert reason.startswith("ml-unavailable:")
