from __future__ import annotations

from dataclasses import dataclass

from trading_algo.core import BUY, SELL
from trading_algo.strategy import PositionState


def _pnl_dollars(position: PositionState, price: float, tick_size: float, tick_value: float) -> float:
    if not position.in_position:
        return 0.0
    if position.side not in (BUY, SELL):
        return 0.0
    if position.entry_price is None:
        return 0.0
    if position.size <= 0:
        return 0.0
    if tick_size <= 0 or tick_value <= 0:
        return 0.0

    ticks = (price - float(position.entry_price)) / tick_size
    if position.side == SELL:
        ticks = -ticks
    return ticks * tick_value * position.size


@dataclass(frozen=True)
class DrawdownSnapshot:
    realized_pnl: float
    unrealized_pnl: float
    equity_pnl: float
    peak_pnl: float
    drawdown_abs: float
    breached: bool


class DrawdownGuard:
    """
    Tracks strategy-level drawdown from the local runtime stream.
    The baseline is 0 PnL; drawdown is measured from peak floating/realized PnL.
    """

    def __init__(self, max_drawdown_abs: float, tick_size: float, tick_value: float, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.max_drawdown_abs = max(0.0, float(max_drawdown_abs))
        self.tick_size = float(tick_size)
        self.tick_value = float(tick_value)

        self._realized_pnl = 0.0
        self._peak_pnl = 0.0
        self._prev_position = PositionState(in_position=False)

    def update(self, position: PositionState, price: float) -> DrawdownSnapshot:
        if not self.enabled or self.max_drawdown_abs <= 0:
            return DrawdownSnapshot(
                realized_pnl=self._realized_pnl,
                unrealized_pnl=0.0,
                equity_pnl=self._realized_pnl,
                peak_pnl=self._peak_pnl,
                drawdown_abs=0.0,
                breached=False,
            )

        prev = self._prev_position
        if prev.in_position and (not position.in_position):
            # Position closed since last update.
            self._realized_pnl += _pnl_dollars(prev, price, self.tick_size, self.tick_value)
        elif prev.in_position and position.in_position:
            # Account for reversals/size changes as close+reopen.
            if (
                prev.side != position.side
                or prev.size != position.size
                or prev.entry_price != position.entry_price
            ):
                self._realized_pnl += _pnl_dollars(prev, price, self.tick_size, self.tick_value)

        unrealized = _pnl_dollars(position, price, self.tick_size, self.tick_value)
        equity = self._realized_pnl + unrealized
        if equity > self._peak_pnl:
            self._peak_pnl = equity
        drawdown = max(0.0, self._peak_pnl - equity)
        breached = drawdown >= self.max_drawdown_abs
        self._prev_position = position

        return DrawdownSnapshot(
            realized_pnl=self._realized_pnl,
            unrealized_pnl=unrealized,
            equity_pnl=equity,
            peak_pnl=self._peak_pnl,
            drawdown_abs=drawdown,
            breached=breached,
        )

