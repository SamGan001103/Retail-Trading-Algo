from __future__ import annotations

from dataclasses import dataclass

from trading_algo.execution import BUY
from trading_algo.strategy.base import MarketBar, PositionState, Strategy, StrategyContext, StrategyDecision


@dataclass(frozen=True)
class OneShotLongStrategy(Strategy):
    hold_bars: int = 20
    size: int = 1

    def on_bar(self, bar: MarketBar, context: StrategyContext, position: PositionState) -> StrategyDecision:  # noqa: ARG002
        if not position.in_position and context.index > 0:
            return StrategyDecision(
                should_exit=False,
                should_enter=True,
                side=BUY,
                size=self.size,
                reason="oneshot-entry",
            )

        if position.in_position and position.bars_in_position >= self.hold_bars:
            return StrategyDecision(
                should_exit=True,
                should_enter=False,
                side=position.side if position.side is not None else BUY,
                size=position.size,
                reason="oneshot-time-exit",
            )

        return StrategyDecision(
            should_exit=False,
            should_enter=False,
            side=position.side if position.side is not None else BUY,
            size=position.size if position.size > 0 else self.size,
            reason="hold",
        )

