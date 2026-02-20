from pathlib import Path

from trading_algo.backtest import BacktestConfig, OrderFlowTick, load_orderflow_ticks_from_csv, run_backtest_orderflow
from trading_algo.core import BUY
from trading_algo.strategy.base import PositionState, StrategyContext, StrategyDecision


class _TickEnterExitStrategy:
    def __init__(self) -> None:
        self._armed = True

    def set_orderflow_state(self, flow) -> None:  # noqa: ANN001
        self._last_flow = flow

    def pending_setup(self):
        return {"armed": True} if self._armed else None

    def on_tick(self, ts: str, price: float, context: StrategyContext, position: PositionState) -> StrategyDecision:  # noqa: ARG002
        if not position.in_position:
            self._armed = False
            return StrategyDecision(
                should_exit=False,
                should_enter=True,
                side=BUY,
                size=1,
                reason="tick-enter",
                sl_ticks_abs=4,
                tp_ticks_abs=8,
            )
        return StrategyDecision(
            should_exit=True,
            should_enter=False,
            side=BUY,
            size=1,
            reason="tick-exit",
        )

    def on_bar(self, bar, context: StrategyContext, position: PositionState) -> StrategyDecision:  # noqa: ANN001, ARG002
        return StrategyDecision(
            should_exit=False,
            should_enter=False,
            side=BUY,
            size=1,
            reason="hold",
        )


def test_load_orderflow_ticks_from_csv(tmp_path: Path):
    csv_path = tmp_path / "ticks.csv"
    csv_path.write_text(
        "timestamp,seq,price,size,bid,ask,bid_size,ask_size\n"
        "2026-01-01T00:00:00Z,7,100.0,2,99.75,100.25,12,8\n",
        encoding="utf-8",
    )
    ticks = load_orderflow_ticks_from_csv(str(csv_path))
    assert len(ticks) == 1
    tick = ticks[0]
    assert tick.price == 100.0
    assert tick.seq == 7
    assert tick.depth is not None
    assert tick.depth.get("bestBidSize") == 12.0
    assert tick.depth.get("bestAskSize") == 8.0


def test_run_backtest_orderflow_executes_tick_entry_and_exit():
    strategy = _TickEnterExitStrategy()
    ticks = [
        OrderFlowTick(
            ts="2026-01-01T00:00:00Z",
            price=100.0,
            volume=1.0,
            quote={"bid": 99.75, "ask": 100.25},
            trade={"price": 100.0, "size": 1.0},
            depth={"bestBidSize": 12.0, "bestAskSize": 8.0},
        ),
        OrderFlowTick(
            ts="2026-01-01T00:00:01Z",
            price=101.0,
            volume=1.0,
            quote={"bid": 100.75, "ask": 101.25},
            trade={"price": 101.0, "size": 1.0},
            depth={"bestBidSize": 9.0, "bestAskSize": 11.0},
        ),
    ]
    result = run_backtest_orderflow(
        ticks,
        strategy,
        BacktestConfig(initial_cash=10_000, fee_per_order=0.0, slippage_bps=0.0, tick_size=0.25),
        bar_sec=60,
    )
    assert result.num_trades == 1
    trade = result.trades[0]
    # Side-correct marketable fills: long entry at ask, long exit at bid.
    assert trade.entry_price == 100.25
    assert trade.exit_price == 100.75
    assert trade.pnl == 0.5


class _EnterHoldExitStrategy:
    def __init__(self) -> None:
        self._armed = True
        self._ticks = 0

    def set_orderflow_state(self, flow) -> None:  # noqa: ANN001
        self._last_flow = flow

    def pending_setup(self):
        return {"armed": True} if self._armed else None

    def on_tick(self, ts: str, price: float, context: StrategyContext, position: PositionState) -> StrategyDecision:  # noqa: ARG002
        self._ticks += 1
        if not position.in_position and self._armed:
            self._armed = False
            return StrategyDecision(
                should_exit=False,
                should_enter=True,
                side=BUY,
                size=1,
                reason="tick-enter",
                sl_ticks_abs=4,
                tp_ticks_abs=40,
            )
        if position.in_position and self._ticks >= 3:
            return StrategyDecision(
                should_exit=True,
                should_enter=False,
                side=BUY,
                size=1,
                reason="tick-exit",
            )
        return StrategyDecision(
            should_exit=False,
            should_enter=False,
            side=BUY,
            size=1,
            reason="hold",
        )

    def on_bar(self, bar, context: StrategyContext, position: PositionState) -> StrategyDecision:  # noqa: ANN001, ARG002
        return StrategyDecision(
            should_exit=False,
            should_enter=False,
            side=BUY,
            size=1,
            reason="hold",
        )


def test_side_correct_stop_does_not_trigger_on_last_only_cross():
    strategy = _EnterHoldExitStrategy()
    executions: list[dict] = []
    ticks = [
        # Entry should fill at ask 101.0, SL at 100.0.
        OrderFlowTick(
            ts="2026-01-01T00:00:00Z",
            price=100.5,
            volume=1.0,
            quote={"bid": 100.5, "ask": 101.0},
            trade={"price": 100.5, "size": 1.0},
            depth={"bestBidSize": 12.0, "bestAskSize": 8.0},
            seq=1,
        ),
        # Last trades below stop, but bid is still above stop => no SL trigger.
        OrderFlowTick(
            ts="2026-01-01T00:00:01Z",
            price=99.5,
            volume=1.0,
            quote={"bid": 100.25, "ask": 100.75},
            trade={"price": 99.5, "size": 1.0},
            depth={"bestBidSize": 9.0, "bestAskSize": 11.0},
            seq=2,
        ),
        # Exit by strategy.
        OrderFlowTick(
            ts="2026-01-01T00:00:02Z",
            price=101.5,
            volume=1.0,
            quote={"bid": 101.25, "ask": 101.5},
            trade={"price": 101.5, "size": 1.0},
            depth={"bestBidSize": 9.0, "bestAskSize": 11.0},
            seq=3,
        ),
    ]
    result = run_backtest_orderflow(
        ticks,
        strategy,
        BacktestConfig(initial_cash=10_000, fee_per_order=0.0, slippage_bps=0.0, tick_size=0.25),
        bar_sec=60,
        execution_callback=lambda e: executions.append(e),
    )
    assert result.num_trades == 1
    assert not any(e.get("event_name") == "protective_exit_tick" for e in executions)
    trade = result.trades[0]
    assert trade.entry_price == 101.0
    assert trade.exit_price == 101.25


def test_replay_sorts_same_timestamp_by_seq():
    strategy = _TickEnterExitStrategy()
    ticks = [
        # Out-of-order input: this should become second after sort.
        OrderFlowTick(
            ts="2026-01-01T00:00:00Z",
            price=100.0,
            volume=1.0,
            quote={"bid": 102.75, "ask": 103.0},
            trade={"price": 100.0, "size": 1.0},
            depth={"bestBidSize": 12.0, "bestAskSize": 8.0},
            seq=2,
        ),
        # This should replay first.
        OrderFlowTick(
            ts="2026-01-01T00:00:00Z",
            price=100.0,
            volume=1.0,
            quote={"bid": 99.75, "ask": 100.25},
            trade={"price": 100.0, "size": 1.0},
            depth={"bestBidSize": 12.0, "bestAskSize": 8.0},
            seq=1,
        ),
    ]
    result = run_backtest_orderflow(
        ticks,
        strategy,
        BacktestConfig(initial_cash=10_000, fee_per_order=0.0, slippage_bps=0.0, tick_size=0.25),
        bar_sec=60,
    )
    assert result.num_trades == 1
    trade = result.trades[0]
    # Entry must use seq=1 quote ask, not seq=2 quote ask.
    assert trade.entry_price == 100.25


class _BoundaryBarStrategy:
    def __init__(self) -> None:
        self.bar_closes: list[float] = []

    def on_bar(self, bar, context: StrategyContext, position: PositionState) -> StrategyDecision:  # noqa: ANN001, ARG002
        self.bar_closes.append(bar.close)
        return StrategyDecision(should_exit=False, should_enter=False, side=BUY, size=1, reason="hold")

    def on_tick(self, ts: str, price: float, context: StrategyContext, position: PositionState) -> StrategyDecision:  # noqa: ARG002
        return StrategyDecision(should_exit=False, should_enter=False, side=BUY, size=1, reason="tick-hold")


def test_bar_boundary_event_isolated_to_new_bucket():
    strategy = _BoundaryBarStrategy()
    ticks = [
        OrderFlowTick(
            ts="2026-01-01T00:00:59.900Z",
            price=100.0,
            volume=1.0,
            quote={"bid": 99.75, "ask": 100.25},
            trade={"price": 100.0, "size": 1.0},
            depth={"bestBidSize": 12.0, "bestAskSize": 8.0},
            seq=1,
        ),
        OrderFlowTick(
            ts="2026-01-01T00:01:00.000Z",
            price=110.0,
            volume=1.0,
            quote={"bid": 109.75, "ask": 110.25},
            trade={"price": 110.0, "size": 1.0},
            depth={"bestBidSize": 12.0, "bestAskSize": 8.0},
            seq=2,
        ),
        OrderFlowTick(
            ts="2026-01-01T00:01:00.100Z",
            price=111.0,
            volume=1.0,
            quote={"bid": 110.75, "ask": 111.25},
            trade={"price": 111.0, "size": 1.0},
            depth={"bestBidSize": 12.0, "bestAskSize": 8.0},
            seq=3,
        ),
    ]
    run_backtest_orderflow(
        ticks,
        strategy,
        BacktestConfig(initial_cash=10_000, fee_per_order=0.0, slippage_bps=0.0, tick_size=0.25),
        bar_sec=60,
    )
    # First completed bar must close with the pre-boundary tick only.
    assert len(strategy.bar_closes) >= 1
    assert strategy.bar_closes[0] == 100.0


class _DelayEntryStrategy:
    def __init__(self) -> None:
        self._armed = True

    def set_orderflow_state(self, flow) -> None:  # noqa: ANN001
        self._last_flow = flow

    def pending_setup(self):
        return {"armed": True}

    def on_tick(self, ts: str, price: float, context: StrategyContext, position: PositionState) -> StrategyDecision:  # noqa: ARG002
        if not position.in_position and self._armed:
            self._armed = False
            return StrategyDecision(
                should_exit=False,
                should_enter=True,
                side=BUY,
                size=1,
                reason="enter-delayed",
                sl_ticks_abs=4,
                tp_ticks_abs=8,
            )
        return StrategyDecision(
            should_exit=False,
            should_enter=False,
            side=BUY,
            size=1,
            reason="hold",
        )

    def on_bar(self, bar, context: StrategyContext, position: PositionState) -> StrategyDecision:  # noqa: ANN001, ARG002
        return StrategyDecision(should_exit=False, should_enter=False, side=BUY, size=1, reason="hold")


def test_next_event_entry_delay_uses_later_quote():
    strategy = _DelayEntryStrategy()
    ticks = [
        OrderFlowTick(
            ts="2026-01-01T00:00:00Z",
            price=100.0,
            volume=1.0,
            quote={"bid": 99.75, "ask": 100.25},
            trade={"price": 100.0, "size": 1.0},
            depth={"bestBidSize": 12.0, "bestAskSize": 8.0},
            seq=1,
        ),
        OrderFlowTick(
            ts="2026-01-01T00:00:01Z",
            price=101.0,
            volume=1.0,
            quote={"bid": 100.75, "ask": 101.25},
            trade={"price": 101.0, "size": 1.0},
            depth={"bestBidSize": 9.0, "bestAskSize": 11.0},
            seq=2,
        ),
    ]
    result = run_backtest_orderflow(
        ticks,
        strategy,
        BacktestConfig(
            initial_cash=10_000,
            fee_per_order=0.0,
            slippage_bps=0.0,
            tick_size=0.25,
            entry_delay_events=1,
        ),
        bar_sec=60,
    )
    assert result.num_trades == 1
    trade = result.trades[0]
    assert trade.entry_ts == "2026-01-01T00:00:01Z"
    assert trade.entry_price == 101.25
