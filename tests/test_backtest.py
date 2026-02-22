from trading_algo.core import BUY
from trading_algo.backtest import BacktestConfig, run_backtest
from trading_algo.strategy import OneShotLongStrategy
from trading_algo.strategy.base import MarketBar, PositionState, StrategyContext, StrategyDecision


class _BracketOnlyStrategy:
    def __init__(self) -> None:
        self._entered = False

    def on_bar(self, bar: MarketBar, context: StrategyContext, position: PositionState) -> StrategyDecision:  # noqa: ARG002
        if not self._entered and not position.in_position:
            self._entered = True
            return StrategyDecision(
                should_exit=False,
                should_enter=True,
                side=BUY,
                size=1,
                reason="enter",
                sl_ticks_abs=4,
                tp_ticks_abs=8,
            )
        return StrategyDecision(should_exit=False, should_enter=False, side=BUY, size=1, reason="hold")


class _AlwaysReenterWithStopStrategy:
    def on_bar(self, bar: MarketBar, context: StrategyContext, position: PositionState) -> StrategyDecision:  # noqa: ARG002
        if position.in_position:
            return StrategyDecision(should_exit=False, should_enter=False, side=BUY, size=1, reason="hold")
        return StrategyDecision(
            should_exit=False,
            should_enter=True,
            side=BUY,
            size=1,
            reason="enter",
            sl_ticks_abs=4,
            tp_ticks_abs=0,
        )


def test_run_backtest_oneshot():
    bars = [
        MarketBar(ts="2026-01-01T00:00:00Z", open=100.0, high=101.0, low=99.0, close=100.0, volume=1000),
        MarketBar(ts="2026-01-01T00:01:00Z", open=100.0, high=101.0, low=99.0, close=101.0, volume=1000),
        MarketBar(ts="2026-01-01T00:02:00Z", open=101.0, high=102.0, low=100.0, close=102.0, volume=1000),
        MarketBar(ts="2026-01-01T00:03:00Z", open=102.0, high=103.0, low=101.0, close=103.0, volume=1000),
        MarketBar(ts="2026-01-01T00:04:00Z", open=103.0, high=104.0, low=102.0, close=104.0, volume=1000),
        MarketBar(ts="2026-01-01T00:05:00Z", open=104.0, high=105.0, low=103.0, close=105.0, volume=1000),
    ]
    strategy = OneShotLongStrategy(hold_bars=2, size=1)
    result = run_backtest(bars, strategy, BacktestConfig(initial_cash=10_000, fee_per_order=0, slippage_bps=0))
    assert result.num_trades >= 1
    assert result.final_equity > 0


def test_backtest_executes_stop_loss_bracket():
    bars = [
        MarketBar(ts="2026-01-01T00:00:00Z", open=100.0, high=100.0, low=100.0, close=100.0, volume=1_000),
        MarketBar(ts="2026-01-01T00:01:00Z", open=100.0, high=100.2, low=98.9, close=100.1, volume=1_000),
    ]
    strategy = _BracketOnlyStrategy()
    result = run_backtest(
        bars,
        strategy,
        BacktestConfig(initial_cash=10_000, fee_per_order=0, slippage_bps=0, tick_size=0.25),
    )
    assert result.num_trades == 1
    trade = result.trades[0]
    assert trade.entry_price == 100.0
    assert trade.exit_price == 99.0
    assert trade.pnl == -1.0


def test_backtest_halts_entries_after_absolute_drawdown_breach():
    bars = [
        MarketBar(ts="2026-01-01T00:00:00Z", open=100.0, high=100.1, low=99.9, close=100.0, volume=1_000),
        MarketBar(ts="2026-01-01T00:01:00Z", open=100.0, high=100.1, low=98.8, close=100.0, volume=1_000),
        MarketBar(ts="2026-01-01T00:02:00Z", open=100.0, high=100.1, low=98.8, close=100.0, volume=1_000),
        MarketBar(ts="2026-01-01T00:03:00Z", open=100.0, high=100.1, low=98.8, close=100.0, volume=1_000),
    ]
    strategy = _AlwaysReenterWithStopStrategy()
    result = run_backtest(
        bars,
        strategy,
        BacktestConfig(
            initial_cash=10_000,
            fee_per_order=0,
            slippage_bps=0,
            tick_size=0.25,
            max_drawdown_abs=2.0,
        ),
    )
    # Each stopped trade loses 1.0, so drawdown guard should stop after two losses.
    assert result.num_trades == 2
