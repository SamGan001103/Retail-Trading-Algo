from trading_algo.core import BUY, SELL
from trading_algo.position_management import StopLossPlanner, TakeProfitPlanner


def test_stop_loss_planner_uses_closest_invalidation_with_noise_buffer():
    planner = StopLossPlanner(tick_size=0.25, noise_buffer_ticks=2, min_stop_ticks=2, max_stop_ticks=200)
    plan = planner.plan(side=BUY, entry_price=21050.0, invalidation_levels=[21044.0, 21047.0, 21020.0])
    assert plan is not None
    assert plan.order_type == "stop_market"
    assert plan.level == 21046.5
    assert plan.ticks == 14


def test_take_profit_planner_filters_by_rrr_and_picks_median_level():
    planner = TakeProfitPlanner(tick_size=0.25, min_rrr=3.0, max_rrr=10.0, front_run_ticks=2)
    plan = planner.plan(
        side=BUY,
        entry_price=21050.0,
        risk_ticks=10,
        target_levels=[21060.0, 21070.0, 21085.0],
    )
    assert plan is not None
    assert plan.order_type == "limit"
    assert plan.level == 21069.5
    assert plan.ticks == 78
    assert 3.0 <= plan.rrr <= 10.0


def test_take_profit_planner_returns_none_when_no_valid_rrr_target():
    planner = TakeProfitPlanner(tick_size=0.25, min_rrr=3.0, max_rrr=10.0, front_run_ticks=2)
    plan = planner.plan(
        side=SELL,
        entry_price=21050.0,
        risk_ticks=20,
        target_levels=[21049.0, 21048.0],
    )
    assert plan is None
