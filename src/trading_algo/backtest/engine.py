from __future__ import annotations

from dataclasses import dataclass

from trading_algo.execution import BUY
from trading_algo.strategy import MarketBar, PositionState, Strategy, StrategyContext


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 10_000.0
    fee_per_order: float = 1.0
    slippage_bps: float = 1.0


@dataclass(frozen=True)
class ExecutedTrade:
    entry_ts: str
    exit_ts: str
    side: int
    size: int
    entry_price: float
    exit_price: float
    pnl: float


@dataclass(frozen=True)
class BacktestResult:
    initial_cash: float
    final_equity: float
    net_pnl: float
    total_return_pct: float
    num_trades: int
    win_rate_pct: float
    max_drawdown_pct: float
    trades: list[ExecutedTrade]


def _apply_slippage(price: float, side: int, slippage_bps: float, is_entry: bool) -> float:
    slip = slippage_bps / 10_000.0
    if side == BUY:
        return price * (1 + slip) if is_entry else price * (1 - slip)
    return price * (1 - slip) if is_entry else price * (1 + slip)


def _trade_pnl(side: int, size: int, entry: float, exit_price: float, fee_per_order: float) -> float:
    gross = (exit_price - entry) * size if side == BUY else (entry - exit_price) * size
    fees = fee_per_order * 2.0
    return gross - fees


def run_backtest(bars: list[MarketBar], strategy: Strategy, config: BacktestConfig) -> BacktestResult:
    cash = float(config.initial_cash)
    equity = cash
    peak_equity = equity
    max_drawdown = 0.0

    in_position = False
    side: int | None = None
    size = 0
    entry_price = 0.0
    entry_ts = ""
    bars_in_position = 0
    trades: list[ExecutedTrade] = []

    for idx, bar in enumerate(bars):
        context = StrategyContext(index=idx, total_bars=len(bars))
        position = PositionState(
            in_position=in_position,
            side=side,
            size=size,
            entry_price=entry_price if in_position else None,
            bars_in_position=bars_in_position if in_position else 0,
        )
        decision = strategy.on_bar(bar, context, position)

        if not in_position and decision.should_enter:
            side = decision.side
            size = max(1, int(decision.size))
            entry_price = _apply_slippage(bar.close, side, config.slippage_bps, is_entry=True)
            entry_ts = bar.ts
            in_position = True
            bars_in_position = 0
        elif in_position and decision.should_exit and side is not None:
            exit_price = _apply_slippage(bar.close, side, config.slippage_bps, is_entry=False)
            pnl = _trade_pnl(side, size, entry_price, exit_price, config.fee_per_order)
            cash += pnl
            trades.append(
                ExecutedTrade(
                    entry_ts=entry_ts,
                    exit_ts=bar.ts,
                    side=side,
                    size=size,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    pnl=pnl,
                )
            )
            in_position = False
            side = None
            size = 0
            entry_price = 0.0
            entry_ts = ""
            bars_in_position = 0
        elif in_position:
            bars_in_position += 1

        if in_position and side is not None:
            unrealized = (bar.close - entry_price) * size if side == BUY else (entry_price - bar.close) * size
            equity = cash + unrealized
        else:
            equity = cash

        peak_equity = max(peak_equity, equity)
        if peak_equity > 0:
            drawdown = (peak_equity - equity) / peak_equity
            max_drawdown = max(max_drawdown, drawdown)

    if in_position and side is not None:
        last = bars[-1]
        exit_price = _apply_slippage(last.close, side, config.slippage_bps, is_entry=False)
        pnl = _trade_pnl(side, size, entry_price, exit_price, config.fee_per_order)
        cash += pnl
        trades.append(
            ExecutedTrade(
                entry_ts=entry_ts,
                exit_ts=last.ts,
                side=side,
                size=size,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl=pnl,
            )
        )
        equity = cash

    wins = sum(1 for t in trades if t.pnl > 0)
    num_trades = len(trades)
    win_rate = (wins / num_trades * 100.0) if num_trades else 0.0
    net_pnl = equity - config.initial_cash
    total_return_pct = (net_pnl / config.initial_cash * 100.0) if config.initial_cash else 0.0

    return BacktestResult(
        initial_cash=config.initial_cash,
        final_equity=equity,
        net_pnl=net_pnl,
        total_return_pct=total_return_pct,
        num_trades=num_trades,
        win_rate_pct=win_rate,
        max_drawdown_pct=max_drawdown * 100.0,
        trades=trades,
    )

