from datetime import datetime, timezone

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
        ml_decision_policy="enforce",
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
    bar_events = strategy.drain_candidate_events()
    assert {e["status"] for e in bar_events} >= {"detected", "armed"}

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
    tick_events = strategy.drain_candidate_events()
    assert {e["status"] for e in tick_events} >= {"entered"}


def test_tick_mode_setup_is_rejected_when_ml_gate_denies():
    class RejectGate:
        def evaluate(self, setup):
            return False, 0.1, "reject"

    strategy = NYSessionMarketStructureStrategy(
        entry_mode="tick",
        require_orderflow=False,
        min_confluence_score=1,
        ml_decision_policy="enforce",
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
    events = strategy.drain_candidate_events()
    assert any(e.get("status") == "rejected" and "ml-reject" in str(e.get("reason")) for e in events)


def test_tick_mode_ml_policy_off_bypasses_rejecting_gate():
    class RejectGate:
        def evaluate(self, setup):
            return False, 0.1, "reject"

    class _Plan:
        def __init__(self, setup):
            self.setup = setup
            self.size = 1
            self.sl_ticks_abs = 10
            self.tp_ticks_abs = 30
            self.ml_score = None

    strategy = NYSessionMarketStructureStrategy(
        entry_mode="tick",
        require_orderflow=False,
        min_confluence_score=1,
        ml_decision_policy="off",
    )
    strategy.set_ml_gate(RejectGate())
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
    strategy._build_entry_risk_plan = lambda setup, **kwargs: _Plan(setup)  # type: ignore[attr-defined]

    decision = strategy.on_bar(
        bar=_bar("2026-01-15T14:31:00Z", 100, 101, 99, 100),
        context=StrategyContext(index=10, total_bars=100),
        position=PositionState(in_position=False),
    )
    assert decision.should_enter is False
    assert decision.reason == "setup-armed-tick"
    assert strategy.pending_setup() is not None
    events = strategy.drain_candidate_events()
    assert any(e.get("status") == "armed" for e in events)
    assert not any("ml-reject" in str(e.get("reason")) for e in events)


def test_tick_mode_news_blackout_invalidates_pending_setup():
    class _Plan:
        def __init__(self, setup):
            self.setup = setup
            self.size = 1
            self.sl_ticks_abs = 10
            self.tp_ticks_abs = 30
            self.ml_score = None

    strategy = NYSessionMarketStructureStrategy(
        entry_mode="tick",
        require_orderflow=False,
        min_confluence_score=1,
        avoid_news=True,
        news_blackouts_utc=[
            (
                datetime(2026, 1, 15, 14, 31, 20, tzinfo=timezone.utc),
                datetime(2026, 1, 15, 14, 31, 40, tzinfo=timezone.utc),
            )
        ],
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
    assert bar_decision.reason == "setup-armed-tick"
    assert strategy.pending_setup() is not None

    tick_decision = strategy.on_tick(
        ts="2026-01-15T14:31:30Z",
        price=100.1,
        context=StrategyContext(index=10, total_bars=100),
        position=PositionState(in_position=False),
    )
    assert tick_decision.should_enter is False
    assert tick_decision.reason == "tick-news-blackout"
    assert strategy.pending_setup() is None
    events = strategy.drain_candidate_events()
    assert any(e.get("status") == "invalidated" and e.get("reason") == "news-blackout" for e in events)


def test_news_blackout_half_open_allows_exact_end_timestamp():
    strategy = NYSessionMarketStructureStrategy(
        avoid_news=True,
        news_blackouts_utc=[
            (
                datetime(2026, 1, 15, 14, 31, 20, tzinfo=timezone.utc),
                datetime(2026, 1, 15, 14, 31, 40, tzinfo=timezone.utc),
            )
        ],
    )
    assert strategy._is_news_blackout("2026-01-15T14:31:39Z") is True  # type: ignore[attr-defined]
    assert strategy._is_news_blackout("2026-01-15T14:31:40Z") is False  # type: ignore[attr-defined]
