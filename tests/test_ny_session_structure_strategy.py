from trading_algo.core import BUY, SELL
from trading_algo.strategy import MarketBar, NYSessionMarketStructureStrategy, PositionState, SetupEnvironment, StrategyContext


def _bar(ts: str, o: float, h: float, l: float, c: float, v: float = 1.0) -> MarketBar:
    return MarketBar(ts=ts, open=o, high=h, low=l, close=c, volume=v)


def test_ny_session_window_membership():
    strategy = NYSessionMarketStructureStrategy()
    # 14:30 UTC = 09:30 America/New_York (EST)
    assert strategy.is_in_session("2026-01-15T14:30:00Z") is True
    # 21:30 UTC = 16:30 America/New_York (outside by default window)
    assert strategy.is_in_session("2026-01-15T21:30:00Z") is False


def test_outside_session_flat_no_entry():
    strategy = NYSessionMarketStructureStrategy(
        htf_aggregation=1,
        htf_swing_strength_high=1,
        htf_swing_strength_low=1,
        ltf_swing_strength_high=1,
        ltf_swing_strength_low=1,
        min_confluence_score=0,
        require_orderflow=False,
    )
    decision = strategy.on_bar(
        bar=_bar("2026-01-15T21:30:00Z", 100, 101, 99, 100),
        context=StrategyContext(index=0, total_bars=100),
        position=PositionState(in_position=False),
    )
    assert decision.should_enter is False
    assert decision.reason == "flat-outside-session"


def test_tick_mode_arms_setup_on_bar_then_enters_on_tick():
    class _Plan:
        def __init__(self, setup):
            self.setup = setup
            self.size = 2
            self.sl_ticks_abs = 12
            self.tp_ticks_abs = 48
            self.ml_score = 0.9

    strategy = NYSessionMarketStructureStrategy(
        entry_mode="tick",
        require_orderflow=False,
        min_confluence_score=1,
    )
    strategy._build_setup_environment = lambda *args, **kwargs: SetupEnvironment(  # type: ignore[attr-defined]
        side=kwargs["side"],
        index=kwargs["context"].index,
        ts=kwargs["bar"].ts,
        close=kwargs["bar"].close,
        has_recent_sweep=True,
        htf_bias="bullish",
        bias_ok=True,
        continuation=True,
        reversal=False,
        equal_levels=True,
        fib_retracement=False,
        key_area_proximity=False,
        confluence_score=1,
    )
    strategy._setup_ready = lambda setup: setup.side == BUY  # type: ignore[attr-defined]
    strategy._tick_entry_ready = lambda side: side == BUY  # type: ignore[attr-defined]
    strategy._build_entry_risk_plan = lambda setup, **kwargs: _Plan(setup)  # type: ignore[attr-defined]

    bar_decision = strategy.on_bar(
        bar=_bar("2026-01-15T14:31:00Z", 100, 101, 99, 100),
        context=StrategyContext(index=10, total_bars=100),
        position=PositionState(in_position=False),
    )
    assert bar_decision.should_enter is False
    assert bar_decision.reason == "setup-armed-tick"
    assert strategy.pending_setup() is not None

    tick_decision = strategy.on_tick(
        ts="2026-01-15T14:31:01Z",
        price=100.25,
        context=StrategyContext(index=10, total_bars=100),
        position=PositionState(in_position=False),
    )
    assert tick_decision.should_enter is True
    assert tick_decision.side == BUY
    assert tick_decision.size == 2
    assert tick_decision.sl_ticks_abs == 12
    assert tick_decision.tp_ticks_abs == 48
    assert tick_decision.reason == "entry-orderflow-sniper"
    assert strategy.pending_setup() is None


def test_tick_mode_setup_is_rejected_when_ml_gate_denies():
    class RejectGate:
        def evaluate(self, setup):
            return False, 0.1, "reject"

    strategy = NYSessionMarketStructureStrategy(
        entry_mode="tick",
        require_orderflow=False,
        min_confluence_score=1,
    )
    strategy.set_ml_gate(RejectGate())
    strategy._build_setup_environment = lambda *args, **kwargs: SetupEnvironment(  # type: ignore[attr-defined]
        side=kwargs["side"],
        index=kwargs["context"].index,
        ts=kwargs["bar"].ts,
        close=kwargs["bar"].close,
        has_recent_sweep=True,
        htf_bias="bearish",
        bias_ok=True,
        continuation=True,
        reversal=False,
        equal_levels=True,
        fib_retracement=False,
        key_area_proximity=False,
        confluence_score=1,
    )
    strategy._setup_ready = lambda setup: setup.side == SELL  # type: ignore[attr-defined]

    decision = strategy.on_bar(
        bar=_bar("2026-01-15T14:31:00Z", 100, 101, 99, 100),
        context=StrategyContext(index=10, total_bars=100),
        position=PositionState(in_position=False),
    )
    assert decision.should_enter is False
    assert decision.reason.startswith("flat-ml-reject:")
    assert strategy.pending_setup() is None
