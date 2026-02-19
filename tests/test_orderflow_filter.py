from trading_algo.core import BUY, SELL
from trading_algo.strategy import (
    DepthImbalanceOrderFlowFilter,
    MarketBar,
    OrderFlowState,
    StrategyContext,
)


def _bar() -> MarketBar:
    return MarketBar(ts="2026-01-15T14:30:00Z", open=100, high=101, low=99, close=100, volume=1)


def test_orderflow_imbalance_from_depth_levels():
    flow = OrderFlowState(
        depth={
            "bids": [{"price": 100.0, "size": 30.0}],
            "asks": [{"price": 100.25, "size": 10.0}],
        }
    )
    imbalance = flow.imbalance()
    assert imbalance is not None
    assert round(imbalance, 4) == 0.5


def test_depth_imbalance_filter_long_short():
    filter_ = DepthImbalanceOrderFlowFilter(min_abs_imbalance=0.2)
    ctx = StrategyContext(index=10, total_bars=100)

    long_flow = OrderFlowState(depth={"bids": [{"size": 40}], "asks": [{"size": 10}]})
    short_flow = OrderFlowState(depth={"bids": [{"size": 10}], "asks": [{"size": 40}]})
    neutral_flow = OrderFlowState(depth={"bids": [{"size": 10}], "asks": [{"size": 10}]})

    assert filter_.allow_entry(BUY, _bar(), ctx, long_flow) is True
    assert filter_.allow_entry(SELL, _bar(), ctx, short_flow) is True
    assert filter_.allow_entry(BUY, _bar(), ctx, neutral_flow) is False
