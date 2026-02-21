from datetime import datetime, time, timezone

from trading_algo.runtime.mode_runner import _in_session_utc, _strategy_from_name
from trading_algo.strategy import NYSessionMarketStructureStrategy


def test_backtest_ny_structure_forces_tick_sniper_orderflow(monkeypatch):
    monkeypatch.setenv("STRAT_ENTRY_MODE", "tick")
    monkeypatch.setenv("STRAT_REQUIRE_ORDERFLOW", "false")
    monkeypatch.setenv("STRAT_BIAS_AGGREGATION", "60")

    strategy = _strategy_from_name(
        "ny_structure",
        hold_bars=120,
        for_forward=False,
        backtest_orderflow_sniper=True,
    )

    assert isinstance(strategy, NYSessionMarketStructureStrategy)
    assert strategy.entry_mode == "tick"
    assert strategy.require_orderflow is True
    assert strategy.bias_aggregation == 60


def test_forward_uses_env_entry_mode_and_orderflow(monkeypatch):
    monkeypatch.setenv("STRAT_ENTRY_MODE", "tick")
    monkeypatch.setenv("STRAT_REQUIRE_ORDERFLOW", "true")
    monkeypatch.setenv("STRAT_BIAS_AGGREGATION", "60")

    strategy = _strategy_from_name("ny_structure", hold_bars=120, for_forward=True)

    assert isinstance(strategy, NYSessionMarketStructureStrategy)
    assert strategy.entry_mode == "tick"
    assert strategy.require_orderflow is True
    assert strategy.bias_aggregation == 60


def test_backtest_ignores_ml_gate_env(monkeypatch):
    monkeypatch.setenv("STRAT_ML_GATE_ENABLED", "true")
    monkeypatch.setenv("STRAT_ML_MODEL_PATH", "")

    strategy = _strategy_from_name(
        "ny_structure",
        hold_bars=120,
        for_forward=False,
        backtest_orderflow_sniper=True,
    )

    assert isinstance(strategy, NYSessionMarketStructureStrategy)
    assert strategy._ml_gate is None


def test_forward_respects_ml_gate_env(monkeypatch):
    monkeypatch.setenv("STRAT_ML_GATE_ENABLED", "true")
    monkeypatch.setenv("STRAT_ML_MODEL_PATH", "")

    strategy = _strategy_from_name("ny_structure", hold_bars=120, for_forward=True)

    assert isinstance(strategy, NYSessionMarketStructureStrategy)
    assert strategy._ml_gate is not None


def test_forward_ml_policy_off_disables_gate_even_if_enabled(monkeypatch):
    monkeypatch.setenv("STRAT_ML_GATE_ENABLED", "true")
    monkeypatch.setenv("STRAT_ML_DECISION_POLICY", "off")
    monkeypatch.setenv("STRAT_ML_MODEL_PATH", "")

    strategy = _strategy_from_name("ny_structure", hold_bars=120, for_forward=True)

    assert isinstance(strategy, NYSessionMarketStructureStrategy)
    assert strategy.ml_decision_policy == "off"
    assert strategy._ml_gate is None


def test_mode_runner_in_session_utc_excludes_weekends():
    sat_dt = datetime(2026, 1, 17, 15, 0, tzinfo=timezone.utc)  # Saturday 10:00 ET
    assert _in_session_utc(sat_dt, start=time(9, 30), end=time(16, 0), tz_name="America/New_York") is False
