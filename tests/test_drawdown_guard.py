from trading_algo.core import BUY
from trading_algo.runtime.drawdown_guard import DrawdownGuard
from trading_algo.strategy import PositionState


def test_drawdown_guard_breaches_on_unrealized_loss():
    guard = DrawdownGuard(max_drawdown_abs=50.0, tick_size=0.25, tick_value=0.5, enabled=True)
    pos = PositionState(in_position=True, side=BUY, size=2, entry_price=100.0, bars_in_position=0)
    # Move +100 ticks first -> peak grows.
    guard.update(pos, price=125.0)
    # Then -100 ticks from entry; drawdown from peak is large.
    snap = guard.update(pos, price=75.0)
    assert snap.breached is True
    assert snap.drawdown_abs >= 50.0


def test_drawdown_guard_accumulates_realized_pnl_after_close():
    guard = DrawdownGuard(max_drawdown_abs=500.0, tick_size=0.25, tick_value=0.5, enabled=True)
    opened = PositionState(in_position=True, side=BUY, size=1, entry_price=100.0, bars_in_position=0)
    guard.update(opened, price=100.0)
    flat = PositionState(in_position=False)
    snap = guard.update(flat, price=101.0)
    # +1.0 point = +4 ticks; 4 * 0.5 = +2.0
    assert round(snap.realized_pnl, 6) == 2.0
    assert snap.breached is False

