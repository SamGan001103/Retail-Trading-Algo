from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from trading_algo.core import BUY
from trading_algo.strategy import MarketBar, PositionState, Strategy, StrategyContext


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 10_000.0
    fee_per_order: float = 1.0
    slippage_bps: float = 1.0
    tick_size: float = 0.25
    max_drawdown_abs: float | None = None


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


def _bracket_levels(
    side: int,
    entry_price: float,
    sl_ticks_abs: int | None,
    tp_ticks_abs: int | None,
    tick_size: float,
) -> tuple[float | None, float | None]:
    if tick_size <= 0:
        return None, None
    sl = abs(int(sl_ticks_abs)) if sl_ticks_abs else 0
    tp = abs(int(tp_ticks_abs)) if tp_ticks_abs else 0
    sl_price = None
    tp_price = None
    if side == BUY:
        if sl > 0:
            sl_price = entry_price - sl * tick_size
        if tp > 0:
            tp_price = entry_price + tp * tick_size
    else:
        if sl > 0:
            sl_price = entry_price + sl * tick_size
        if tp > 0:
            tp_price = entry_price - tp * tick_size
    return sl_price, tp_price


def _protective_exit_price(
    side: int,
    bar: MarketBar,
    sl_price: float | None,
    tp_price: float | None,
) -> tuple[float, str] | None:
    if side == BUY:
        sl_hit = sl_price is not None and bar.low <= sl_price
        tp_hit = tp_price is not None and bar.high >= tp_price
        if sl_hit and tp_hit:
            return float(sl_price), "stop-loss"
        if sl_hit:
            return float(sl_price), "stop-loss"
        if tp_hit:
            return float(tp_price), "take-profit"
        return None

    sl_hit = sl_price is not None and bar.high >= sl_price
    tp_hit = tp_price is not None and bar.low <= tp_price
    if sl_hit and tp_hit:
        return float(sl_price), "stop-loss"
    if sl_hit:
        return float(sl_price), "stop-loss"
    if tp_hit:
        return float(tp_price), "take-profit"
    return None


def _strategy_side_name(side: int | None) -> str:
    if side == 0:
        return "buy"
    if side == 1:
        return "sell"
    return "unknown"


def _drain_candidate_events(strategy: Strategy) -> list[dict[str, Any]]:
    drain = getattr(strategy, "drain_candidate_events", None)
    if not callable(drain):
        return []
    try:
        events = drain()
    except Exception:
        return []
    if not isinstance(events, list):
        return []
    out: list[dict[str, Any]] = []
    for event in events:
        if isinstance(event, dict):
            out.append(event)
    return out


def run_backtest(
    bars: list[MarketBar],
    strategy: Strategy,
    config: BacktestConfig,
    *,
    telemetry_callback: Callable[[dict[str, Any]], None] | None = None,
    execution_callback: Callable[[dict[str, Any]], None] | None = None,
    candidate_callback: Callable[[dict[str, Any]], None] | None = None,
) -> BacktestResult:
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
    sl_price: float | None = None
    tp_price: float | None = None
    trading_halted = False
    trades: list[ExecutedTrade] = []

    for idx, bar in enumerate(bars):
        if in_position and side is not None:
            protective = _protective_exit_price(side, bar, sl_price, tp_price)
            if protective is not None:
                protective_price, _reason = protective
                exit_side = side
                exit_size = size
                exit_price = _apply_slippage(protective_price, side, config.slippage_bps, is_entry=False)
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
                sl_price = None
                tp_price = None

                equity = cash
                peak_equity = max(peak_equity, equity)
                if peak_equity > 0:
                    drawdown = (peak_equity - equity) / peak_equity
                    max_drawdown = max(max_drawdown, drawdown)
                if config.max_drawdown_abs is not None and config.max_drawdown_abs > 0:
                    if (peak_equity - equity) >= config.max_drawdown_abs:
                        trading_halted = True
                if execution_callback is not None:
                    execution_callback(
                        {
                            "event_name": "protective_exit",
                            "bar_index": idx,
                            "bar_ts": bar.ts,
                            "side": _strategy_side_name(exit_side),
                            "size": exit_size,
                            "reason": _reason,
                            "exit_price": round(exit_price, 6),
                            "pnl": round(pnl, 6),
                        }
                    )

        if trading_halted:
            continue

        context = StrategyContext(index=idx, total_bars=len(bars))
        position = PositionState(
            in_position=in_position,
            side=side,
            size=size,
            entry_price=entry_price if in_position else None,
            bars_in_position=bars_in_position if in_position else 0,
        )
        decision = strategy.on_bar(bar, context, position)
        for candidate_event in _drain_candidate_events(strategy):
            if candidate_callback is not None:
                event = {
                    "event_name": "strategy_candidate",
                    "bar_index": idx,
                    "bar_ts": bar.ts,
                }
                event.update(candidate_event)
                candidate_callback(event)

        if not in_position and decision.should_enter:
            side = decision.side
            size = max(1, int(decision.size))
            entry_price = _apply_slippage(bar.close, side, config.slippage_bps, is_entry=True)
            entry_ts = bar.ts
            in_position = True
            bars_in_position = 0
            sl_price, tp_price = _bracket_levels(
                side=side,
                entry_price=entry_price,
                sl_ticks_abs=decision.sl_ticks_abs,
                tp_ticks_abs=decision.tp_ticks_abs,
                tick_size=config.tick_size,
            )
            if execution_callback is not None:
                execution_callback(
                    {
                        "event_name": "enter",
                        "bar_index": idx,
                        "bar_ts": bar.ts,
                        "side": _strategy_side_name(side),
                        "size": size,
                        "reason": decision.reason,
                        "entry_price": round(entry_price, 6),
                        "sl_ticks_abs": int(decision.sl_ticks_abs) if decision.sl_ticks_abs else None,
                        "tp_ticks_abs": int(decision.tp_ticks_abs) if decision.tp_ticks_abs else None,
                    }
                )
        elif in_position and decision.should_exit and side is not None:
            exit_side = side
            exit_size = size
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
            sl_price = None
            tp_price = None
            if execution_callback is not None:
                execution_callback(
                    {
                        "event_name": "exit",
                        "bar_index": idx,
                        "bar_ts": bar.ts,
                        "side": _strategy_side_name(exit_side),
                        "size": exit_size,
                        "reason": decision.reason,
                        "exit_price": round(exit_price, 6),
                        "pnl": round(pnl, 6),
                    }
                )
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
        if config.max_drawdown_abs is not None and config.max_drawdown_abs > 0:
            if (peak_equity - equity) >= config.max_drawdown_abs:
                trading_halted = True
                if in_position and side is not None:
                    exit_side = side
                    exit_size = size
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
                    sl_price = None
                    tp_price = None
                    equity = cash
                    if execution_callback is not None:
                        execution_callback(
                            {
                                "event_name": "halt_exit",
                                "bar_index": idx,
                                "bar_ts": bar.ts,
                                "side": _strategy_side_name(exit_side),
                                "size": exit_size,
                                "reason": "max_drawdown_abs",
                                "exit_price": round(exit_price, 6),
                                "pnl": round(pnl, 6),
                            }
                        )
        if telemetry_callback is not None:
            telemetry_callback(
                {
                    "event_name": "bar_snapshot",
                    "bar_index": idx,
                    "bar_ts": bar.ts,
                    "cash": round(cash, 6),
                    "equity": round(equity, 6),
                    "peak_equity": round(peak_equity, 6),
                    "drawdown_abs": round(max(0.0, peak_equity - equity), 6),
                    "drawdown_pct": round((max_drawdown * 100.0), 6),
                    "in_position": in_position,
                    "position_side": _strategy_side_name(side),
                    "position_size": size if in_position else 0,
                    "decision_reason": decision.reason,
                }
            )

    if in_position and side is not None:
        last = bars[-1]
        exit_side = side
        exit_size = size
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
        if execution_callback is not None:
            execution_callback(
                {
                    "event_name": "force_close_end_of_data",
                    "bar_index": len(bars) - 1,
                    "bar_ts": last.ts,
                    "side": _strategy_side_name(exit_side),
                    "size": exit_size,
                    "reason": "end_of_data",
                    "exit_price": round(exit_price, 6),
                    "pnl": round(pnl, 6),
                }
            )

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
