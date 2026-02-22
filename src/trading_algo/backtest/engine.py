from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from trading_algo.core import BUY, SELL
from trading_algo.strategy import MarketBar, OrderFlowState, PositionState, Strategy, StrategyContext

from .data import OrderFlowTick


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 10_000.0
    fee_per_order: float = 1.0
    slippage_bps: float = 1.0
    tick_size: float = 0.25
    max_drawdown_abs: float | None = None
    slip_entry_ticks: float = 0.0
    slip_stop_ticks: float = 0.0
    slip_tp_ticks: float = 0.0
    spread_slip_k: float = 0.0
    entry_delay_events: int = 0


@dataclass
class MarketSnapshot:
    ts: str = ""
    bid_px: float | None = None
    bid_sz: float | None = None
    ask_px: float | None = None
    ask_sz: float | None = None
    last_px: float | None = None
    last_sz: float | None = None
    has_quote: bool = False
    has_trade: bool = False

    def mid_px(self) -> float | None:
        if self.bid_px is None or self.ask_px is None:
            return None
        return (self.bid_px + self.ask_px) / 2.0

    def spread_ticks(self, tick_size: float) -> float | None:
        if tick_size <= 0:
            return None
        if self.bid_px is None or self.ask_px is None:
            return None
        return (self.ask_px - self.bid_px) / tick_size


@dataclass(frozen=True)
class EntryIntent:
    side: int
    size: int
    reason: str
    sl_ticks_abs: int | None
    tp_ticks_abs: int | None
    candidate_id: str | None
    event_name: str
    remaining_events: int


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


def _parse_ts(ts: str) -> datetime:
    value = ts.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _bucket_end_ts(bucket: int, bar_sec: int) -> str:
    end_ts = datetime.fromtimestamp((int(bucket) + 1) * int(bar_sec), tz=timezone.utc)
    return end_ts.isoformat().replace("+00:00", "Z")


def _num(payload: dict[str, Any] | None, *keys: str) -> float | None:
    if payload is None:
        return None
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _depth_level_price_size(levels: Any) -> tuple[float | None, float | None]:
    if not isinstance(levels, list) or not levels:
        return None, None
    first = levels[0]
    if not isinstance(first, dict):
        return None, None
    return _num(first, "price", "p"), _num(first, "size", "qty", "q")


def _update_snapshot(snapshot: MarketSnapshot, tick: OrderFlowTick) -> None:
    snapshot.ts = tick.ts

    quote_bid = _num(tick.quote, "bid", "bidPrice", "bestBid")
    quote_ask = _num(tick.quote, "ask", "askPrice", "bestAsk")
    depth_bid = _num(tick.depth, "bestBid", "bidPrice")
    depth_ask = _num(tick.depth, "bestAsk", "askPrice")
    depth_bid_level_px, depth_bid_level_sz = _depth_level_price_size(tick.depth.get("bids") if tick.depth else None)
    depth_ask_level_px, depth_ask_level_sz = _depth_level_price_size(tick.depth.get("asks") if tick.depth else None)
    bid_px = quote_bid if quote_bid is not None else (depth_bid if depth_bid is not None else depth_bid_level_px)
    ask_px = quote_ask if quote_ask is not None else (depth_ask if depth_ask is not None else depth_ask_level_px)
    if bid_px is not None:
        snapshot.bid_px = float(bid_px)
    if ask_px is not None:
        snapshot.ask_px = float(ask_px)
    if bid_px is not None or ask_px is not None:
        snapshot.has_quote = True

    bid_sz = _num(tick.depth, "bestBidSize", "bidSize", "bid_size")
    ask_sz = _num(tick.depth, "bestAskSize", "askSize", "ask_size")
    if bid_sz is None:
        bid_sz = depth_bid_level_sz
    if ask_sz is None:
        ask_sz = depth_ask_level_sz
    if bid_sz is not None:
        snapshot.bid_sz = float(bid_sz)
    if ask_sz is not None:
        snapshot.ask_sz = float(ask_sz)

    trade_px = _num(tick.trade, "price", "last", "lastPrice", "tradePrice", "close")
    trade_sz = _num(tick.trade, "size", "qty", "quantity", "volume", "lastSize")
    if trade_px is None:
        trade_px = float(tick.price)
    if trade_sz is None:
        trade_sz = float(tick.volume)
    if trade_px is not None:
        snapshot.last_px = float(trade_px)
        snapshot.has_trade = True
    if trade_sz is not None:
        snapshot.last_sz = float(trade_sz)


def _reference_price(snapshot: MarketSnapshot, fallback: float) -> float:
    mid = snapshot.mid_px()
    if mid is not None:
        return float(mid)
    if snapshot.last_px is not None:
        return float(snapshot.last_px)
    return float(fallback)


def _spread_adjusted_slip(base_slip_ticks: float, snapshot: MarketSnapshot, tick_size: float, spread_slip_k: float) -> float:
    spread_ticks = snapshot.spread_ticks(tick_size)
    if spread_ticks is None:
        return float(base_slip_ticks)
    extra = max(0.0, float(spread_ticks) - 1.0)
    return float(base_slip_ticks) + float(spread_slip_k) * extra


def _market_fill_price(
    *,
    order_side: int,
    snapshot: MarketSnapshot,
    tick_size: float,
    base_slip_ticks: float,
    spread_slip_k: float,
    fallback_price: float | None = None,
    allow_last_fallback: bool = False,
) -> float | None:
    slip_ticks = _spread_adjusted_slip(base_slip_ticks, snapshot, tick_size, spread_slip_k)
    if order_side == BUY:
        if snapshot.ask_px is not None:
            return float(snapshot.ask_px + slip_ticks * tick_size)
        if allow_last_fallback and snapshot.last_px is not None:
            return float(snapshot.last_px + slip_ticks * tick_size)
        if allow_last_fallback and fallback_price is not None:
            return float(fallback_price + slip_ticks * tick_size)
        return None

    if snapshot.bid_px is not None:
        return float(snapshot.bid_px - slip_ticks * tick_size)
    if allow_last_fallback and snapshot.last_px is not None:
        return float(snapshot.last_px - slip_ticks * tick_size)
    if allow_last_fallback and fallback_price is not None:
        return float(fallback_price - slip_ticks * tick_size)
    return None


def _protective_trigger_reason(
    side: int,
    snapshot: MarketSnapshot,
    sl_price: float | None,
    tp_price: float | None,
) -> str | None:
    # Conservative: require quote-side touch. Missing quote means no trigger.
    if side == BUY:
        if snapshot.bid_px is None:
            return None
        if sl_price is not None and snapshot.bid_px <= sl_price:
            return "stop-loss"
        if tp_price is not None and snapshot.bid_px >= tp_price:
            return "take-profit"
        return None

    if snapshot.ask_px is None:
        return None
    if sl_price is not None and snapshot.ask_px >= sl_price:
        return "stop-loss"
    if tp_price is not None and snapshot.ask_px <= tp_price:
        return "take-profit"
    return None


def _safe_set_orderflow_state(strategy: Strategy, tick: OrderFlowTick) -> None:
    setter = getattr(strategy, "set_orderflow_state", None)
    if not callable(setter):
        return
    try:
        setter(OrderFlowState(quote=tick.quote, trade=tick.trade, depth=tick.depth))
    except Exception:
        return


def _safe_pending_setup(strategy: Strategy) -> bool:
    getter = getattr(strategy, "pending_setup", None)
    if not callable(getter):
        return False
    try:
        return getter() is not None
    except Exception:
        return False


def _emit_candidates(
    strategy: Strategy,
    candidate_callback: Callable[[dict[str, Any]], None] | None,
    *,
    source: str,
    bar_index: int,
    ts: str,
) -> list[dict[str, Any]]:
    emitted: list[dict[str, Any]] = []
    for candidate_event in _drain_candidate_events(strategy):
        event = {
            "event_name": "strategy_candidate",
            "source": source,
            "bar_index": bar_index,
            "bar_ts": ts,
        }
        event.update(candidate_event)
        emitted.append(event)
        if candidate_callback is not None:
            candidate_callback(event)
    return emitted


def _entered_candidate_id(events: list[dict[str, Any]]) -> str | None:
    for event in reversed(events):
        if str(event.get("status") or "").strip().lower() != "entered":
            continue
        candidate_id = event.get("candidate_id")
        if candidate_id is not None and str(candidate_id).strip() != "":
            return str(candidate_id)
    return None


def _mark_to_equity(cash: float, in_position: bool, side: int | None, size: int, entry_price: float, price: float) -> float:
    if in_position and side is not None:
        unrealized = (price - entry_price) * size if side == BUY else (entry_price - price) * size
        return cash + unrealized
    return cash


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
    active_candidate_id: str | None = None

    for idx, bar in enumerate(bars):
        if in_position and side is not None:
            protective = _protective_exit_price(side, bar, sl_price, tp_price)
            if protective is not None:
                protective_price, _reason = protective
                exit_side = side
                exit_size = size
                exit_candidate_id = active_candidate_id
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
                active_candidate_id = None

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
                            "candidate_id": exit_candidate_id,
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
        candidate_events = _emit_candidates(
            strategy,
            candidate_callback,
            source="bar",
            bar_index=idx,
            ts=bar.ts,
        )
        entered_candidate_id = _entered_candidate_id(candidate_events)

        if not in_position and decision.should_enter:
            side = decision.side
            size = max(1, int(decision.size))
            entry_price = _apply_slippage(bar.close, side, config.slippage_bps, is_entry=True)
            entry_ts = bar.ts
            in_position = True
            bars_in_position = 0
            active_candidate_id = entered_candidate_id
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
                        "candidate_id": active_candidate_id,
                    }
                )
        elif in_position and decision.should_exit and side is not None:
            exit_side = side
            exit_size = size
            exit_candidate_id = active_candidate_id
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
            active_candidate_id = None
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
                        "candidate_id": exit_candidate_id,
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
                    exit_candidate_id = active_candidate_id
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
                    active_candidate_id = None
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
                                "candidate_id": exit_candidate_id,
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
        exit_candidate_id = active_candidate_id
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
                    "candidate_id": exit_candidate_id,
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


def run_backtest_orderflow(
    ticks: Iterable[OrderFlowTick],
    strategy: Strategy,
    config: BacktestConfig,
    *,
    bar_sec: int = 60,
    telemetry_callback: Callable[[dict[str, Any]], None] | None = None,
    execution_callback: Callable[[dict[str, Any]], None] | None = None,
    candidate_callback: Callable[[dict[str, Any]], None] | None = None,
) -> BacktestResult:
    replay_iter: Iterator[OrderFlowTick]
    first_tick: OrderFlowTick
    if isinstance(ticks, list):
        if not ticks:
            raise RuntimeError("Orderflow backtest requires at least one tick")

        # Deterministic replay: monotonic (ts, seq) fast-path, otherwise sort once.
        ordered = True
        prev_key: tuple[datetime, int] | None = None
        for i, tick in enumerate(ticks):
            key = (_parse_ts(tick.ts), int(tick.seq) if int(tick.seq) > 0 else i + 1)
            if prev_key is not None and key < prev_key:
                ordered = False
                break
            prev_key = key
        replay_ticks = ticks
        if not ordered:
            replay_ticks = sorted(
                ticks,
                key=lambda t: (_parse_ts(t.ts), int(t.seq) if int(t.seq) > 0 else 0),
            )
        first_tick = replay_ticks[0]
        replay_iter = iter(replay_ticks)
    else:
        iterator = iter(ticks)
        try:
            first_tick = next(iterator)
        except StopIteration as exc:
            raise RuntimeError("Orderflow backtest requires at least one tick") from exc

        def _prepend(first: OrderFlowTick, rest: Iterator[OrderFlowTick]) -> Iterator[OrderFlowTick]:
            yield first
            for item in rest:
                yield item

        replay_iter = _prepend(first_tick, iterator)

    bar_sec = max(1, int(bar_sec))

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
    active_candidate_id: str | None = None
    pending_entry: EntryIntent | None = None
    missing_quote_trigger_count = 0
    next_order_id = 1
    next_position_id = 1
    open_position_id: str | None = None

    current_bucket: int | None = None
    bar_open: float | None = None
    bar_high: float | None = None
    bar_low: float | None = None
    bar_close: float | None = None
    bar_volume = 0.0
    bar_ts = ""
    bar_index = 0
    snapshot = MarketSnapshot(ts=first_tick.ts, last_px=float(first_tick.price), last_sz=float(first_tick.volume))
    last_tick_price = first_tick.price
    last_tick_ts = first_tick.ts
    last_tick_idx = 0

    def _close_position(
        *,
        realized_exit: float,
        exit_ts: str,
        reason: str,
        event_name: str,
        idx: int,
    ) -> None:
        nonlocal cash, in_position, side, size, entry_price, entry_ts, bars_in_position, sl_price, tp_price, equity
        nonlocal active_candidate_id, next_order_id, open_position_id
        if not in_position or side is None:
            return
        exit_side = side
        exit_size = size
        exit_candidate_id = active_candidate_id
        exit_order_id = f"bt-order-{next_order_id}"
        next_order_id += 1
        pnl = _trade_pnl(side, size, entry_price, realized_exit, config.fee_per_order)
        cash += pnl
        trades.append(
            ExecutedTrade(
                entry_ts=entry_ts,
                exit_ts=exit_ts,
                side=side,
                size=size,
                entry_price=entry_price,
                exit_price=realized_exit,
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
        active_candidate_id = None
        position_id = open_position_id
        open_position_id = None
        equity = cash
        if execution_callback is not None:
            execution_callback(
                {
                    "event_name": event_name,
                    "bar_index": idx,
                    "bar_ts": exit_ts,
                    "side": _strategy_side_name(exit_side),
                    "size": exit_size,
                    "reason": reason,
                    "exit_price": round(realized_exit, 6),
                    "pnl": round(pnl, 6),
                    "candidate_id": exit_candidate_id,
                    "order_id": exit_order_id,
                    "position_id": position_id,
                }
            )

    def _enter_position(
        *,
        decision_side: int,
        decision_size: int,
        decision_reason: str,
        decision_sl_ticks_abs: int | None,
        decision_tp_ticks_abs: int | None,
        realized_entry: float,
        ts: str,
        idx: int,
        event_name: str,
        candidate_id: str | None,
    ) -> None:
        nonlocal in_position, side, size, entry_price, entry_ts, bars_in_position, sl_price, tp_price
        nonlocal active_candidate_id, next_order_id, next_position_id, open_position_id
        if in_position:
            return
        entry_order_id = f"bt-order-{next_order_id}"
        next_order_id += 1
        position_id = f"bt-position-{next_position_id}"
        next_position_id += 1
        side = decision_side
        size = max(1, int(decision_size))
        entry_price = float(realized_entry)
        entry_ts = ts
        in_position = True
        bars_in_position = 0
        active_candidate_id = candidate_id
        open_position_id = position_id
        sl_price, tp_price = _bracket_levels(
            side=side,
            entry_price=entry_price,
            sl_ticks_abs=decision_sl_ticks_abs,
            tp_ticks_abs=decision_tp_ticks_abs,
            tick_size=config.tick_size,
        )
        if execution_callback is not None:
            execution_callback(
                {
                    "event_name": event_name,
                    "bar_index": idx,
                    "bar_ts": ts,
                    "side": _strategy_side_name(side),
                    "size": size,
                    "reason": decision_reason,
                    "entry_price": round(entry_price, 6),
                    "sl_ticks_abs": int(decision_sl_ticks_abs) if decision_sl_ticks_abs else None,
                    "tp_ticks_abs": int(decision_tp_ticks_abs) if decision_tp_ticks_abs else None,
                    "candidate_id": active_candidate_id,
                    "order_id": entry_order_id,
                    "position_id": position_id,
                }
            )

    def _try_execute_pending(ts: str, idx: int, fallback_price: float) -> bool:
        nonlocal pending_entry
        if pending_entry is None or in_position or trading_halted:
            return False
        if pending_entry.remaining_events > 0:
            return False
        fill_price = _market_fill_price(
            order_side=pending_entry.side,
            snapshot=snapshot,
            tick_size=config.tick_size,
            base_slip_ticks=config.slip_entry_ticks,
            spread_slip_k=config.spread_slip_k,
            fallback_price=fallback_price,
            allow_last_fallback=False,
        )
        if fill_price is None:
            return False
        _enter_position(
            decision_side=pending_entry.side,
            decision_size=pending_entry.size,
            decision_reason=pending_entry.reason,
            decision_sl_ticks_abs=pending_entry.sl_ticks_abs,
            decision_tp_ticks_abs=pending_entry.tp_ticks_abs,
            realized_entry=float(fill_price),
            ts=ts,
            idx=idx,
            event_name=pending_entry.event_name,
            candidate_id=pending_entry.candidate_id,
        )
        pending_entry = None
        return True

    def _process_completed_bar(idx: int) -> None:
        nonlocal bar_index, bars_in_position, trading_halted, peak_equity, max_drawdown
        nonlocal bar_open, bar_high, bar_low, bar_close, bar_volume, bar_ts
        nonlocal pending_entry
        assert bar_open is not None and bar_high is not None and bar_low is not None and bar_close is not None
        completed = MarketBar(
            ts=bar_ts,
            open=bar_open,
            high=bar_high,
            low=bar_low,
            close=bar_close,
            volume=bar_volume,
        )
        context = StrategyContext(index=bar_index, total_bars=bar_index + 1)
        position = PositionState(
            in_position=in_position,
            side=side,
            size=size,
            entry_price=entry_price if in_position else None,
            bars_in_position=bars_in_position if in_position else 0,
        )
        decision = strategy.on_bar(completed, context, position)
        candidate_events = _emit_candidates(strategy, candidate_callback, source="bar", bar_index=bar_index, ts=completed.ts)
        entered_candidate_id = _entered_candidate_id(candidate_events)

        if not trading_halted:
            if decision.should_exit and in_position:
                exit_price = _apply_slippage(completed.close, side if side is not None else BUY, config.slippage_bps, is_entry=False)
                _close_position(
                    realized_exit=exit_price,
                    exit_ts=completed.ts,
                    reason=decision.reason,
                    event_name="exit",
                    idx=idx,
                )
            elif decision.should_enter and not in_position:
                pending = EntryIntent(
                    side=decision.side,
                    size=max(1, int(decision.size)),
                    reason=decision.reason,
                    sl_ticks_abs=decision.sl_ticks_abs,
                    tp_ticks_abs=decision.tp_ticks_abs,
                    candidate_id=entered_candidate_id,
                    event_name="enter",
                    remaining_events=max(0, int(config.entry_delay_events)),
                )
                pending_entry = pending
                fill_ts = snapshot.ts if str(snapshot.ts).strip() != "" else completed.ts
                _try_execute_pending(fill_ts, idx, completed.close)

        if in_position:
            bars_in_position += 1
        else:
            bars_in_position = 0
        bar_index += 1

        marked_equity = _mark_to_equity(cash, in_position, side, size, entry_price, completed.close)
        peak_equity = max(peak_equity, marked_equity)
        if peak_equity > 0:
            drawdown = (peak_equity - marked_equity) / peak_equity
            max_drawdown = max(max_drawdown, drawdown)
        if telemetry_callback is not None:
            telemetry_callback(
                {
                    "event_name": "bar_snapshot",
                    "bar_index": bar_index,
                    "bar_ts": completed.ts,
                    "cash": round(cash, 6),
                    "equity": round(marked_equity, 6),
                    "peak_equity": round(peak_equity, 6),
                    "drawdown_abs": round(max(0.0, peak_equity - marked_equity), 6),
                    "drawdown_pct": round((max_drawdown * 100.0), 6),
                    "in_position": in_position,
                    "position_side": _strategy_side_name(side),
                    "position_size": size if in_position else 0,
                    "decision_reason": decision.reason,
                }
            )

    for idx, tick in enumerate(replay_iter):
        _safe_set_orderflow_state(strategy, tick)
        _update_snapshot(snapshot, tick)
        last_tick_price = tick.price
        last_tick_ts = tick.ts
        last_tick_idx = idx
        tick_dt = _parse_ts(tick.ts)
        bucket = int(tick_dt.timestamp()) // bar_sec

        if current_bucket is None:
            current_bucket = bucket
        elif bucket != current_bucket:
            _process_completed_bar(idx)
            current_bucket = bucket
            bar_open = None
            bar_high = None
            bar_low = None
            bar_close = None
            bar_volume = 0.0

        if bar_open is None:
            bar_open = bar_high = bar_low = bar_close = tick.price
            bar_volume = tick.volume
            bar_ts = _bucket_end_ts(bucket, bar_sec)
        else:
            assert bar_high is not None and bar_low is not None
            bar_high = max(bar_high, tick.price)
            bar_low = min(bar_low, tick.price)
            bar_close = tick.price
            bar_volume += tick.volume

        if in_position and side is not None:
            protective_reason = _protective_trigger_reason(side, snapshot, sl_price, tp_price)
            if protective_reason is None and (snapshot.bid_px is None or snapshot.ask_px is None):
                missing_quote_trigger_count += 1
            if protective_reason is not None:
                order_side = SELL if side == BUY else BUY
                base_slip = config.slip_stop_ticks if protective_reason == "stop-loss" else config.slip_tp_ticks
                fill_price = _market_fill_price(
                    order_side=order_side,
                    snapshot=snapshot,
                    tick_size=config.tick_size,
                    base_slip_ticks=base_slip,
                    spread_slip_k=config.spread_slip_k,
                    fallback_price=tick.price,
                    allow_last_fallback=True,
                )
                if fill_price is not None:
                    _close_position(
                        realized_exit=float(fill_price),
                        exit_ts=tick.ts,
                        reason=protective_reason,
                        event_name="protective_exit_tick",
                        idx=idx,
                    )

        if pending_entry is not None and (not in_position) and (not trading_halted):
            next_remaining = max(0, int(pending_entry.remaining_events) - 1)
            pending_entry = EntryIntent(
                side=pending_entry.side,
                size=pending_entry.size,
                reason=pending_entry.reason,
                sl_ticks_abs=pending_entry.sl_ticks_abs,
                tp_ticks_abs=pending_entry.tp_ticks_abs,
                candidate_id=pending_entry.candidate_id,
                event_name=pending_entry.event_name,
                remaining_events=next_remaining,
            )
            _try_execute_pending(tick.ts, idx, tick.price)

        if not trading_halted:
            context = StrategyContext(index=bar_index, total_bars=bar_index + 1)
            position = PositionState(
                in_position=in_position,
                side=side,
                size=size,
                entry_price=entry_price if in_position else None,
                bars_in_position=bars_in_position if in_position else 0,
            )
            setup_armed = _safe_pending_setup(strategy)
            tick_handler = getattr(strategy, "on_tick", None)
            if callable(tick_handler) and (setup_armed or in_position):
                decision = tick_handler(tick.ts, tick.price, context, position)
                candidate_events = _emit_candidates(strategy, candidate_callback, source="tick", bar_index=bar_index, ts=tick.ts)
                entered_candidate_id = _entered_candidate_id(candidate_events)
                if decision.should_exit and in_position:
                    exit_order_side = SELL if side == BUY else BUY
                    exit_fill = _market_fill_price(
                        order_side=exit_order_side,
                        snapshot=snapshot,
                        tick_size=config.tick_size,
                        base_slip_ticks=config.slip_entry_ticks,
                        spread_slip_k=config.spread_slip_k,
                        fallback_price=tick.price,
                        allow_last_fallback=True,
                    )
                    if exit_fill is not None:
                        _close_position(
                            realized_exit=float(exit_fill),
                            exit_ts=tick.ts,
                            reason=decision.reason,
                            event_name="tick_exit",
                            idx=idx,
                        )
                elif decision.should_enter and (not in_position) and pending_entry is None:
                    pending_entry = EntryIntent(
                        side=decision.side,
                        size=max(1, int(decision.size)),
                        reason=decision.reason,
                        sl_ticks_abs=decision.sl_ticks_abs,
                        tp_ticks_abs=decision.tp_ticks_abs,
                        candidate_id=entered_candidate_id,
                        event_name="tick_enter",
                        remaining_events=max(0, int(config.entry_delay_events)),
                    )
                    _try_execute_pending(tick.ts, idx, tick.price)

        equity = _mark_to_equity(cash, in_position, side, size, entry_price, _reference_price(snapshot, tick.price))
        peak_equity = max(peak_equity, equity)
        if peak_equity > 0:
            drawdown = (peak_equity - equity) / peak_equity
            max_drawdown = max(max_drawdown, drawdown)
        if config.max_drawdown_abs is not None and config.max_drawdown_abs > 0:
            if (peak_equity - equity) >= config.max_drawdown_abs and not trading_halted:
                trading_halted = True
                pending_entry = None
                if in_position:
                    halt_side = SELL if side == BUY else BUY
                    halt_fill = _market_fill_price(
                        order_side=halt_side,
                        snapshot=snapshot,
                        tick_size=config.tick_size,
                        base_slip_ticks=config.slip_entry_ticks,
                        spread_slip_k=config.spread_slip_k,
                        fallback_price=tick.price,
                        allow_last_fallback=True,
                    )
                    if halt_fill is None:
                        continue
                    _close_position(
                        realized_exit=float(halt_fill),
                        exit_ts=tick.ts,
                        reason="max_drawdown_abs",
                        event_name="halt_exit_tick",
                        idx=idx,
                    )

    if current_bucket is not None and bar_open is not None and bar_high is not None and bar_low is not None and bar_close is not None:
        _process_completed_bar(last_tick_idx)

    if in_position and side is not None:
        end_side = SELL if side == BUY else BUY
        end_fill = _market_fill_price(
            order_side=end_side,
            snapshot=snapshot,
            tick_size=config.tick_size,
            base_slip_ticks=config.slip_entry_ticks,
            spread_slip_k=config.spread_slip_k,
            fallback_price=last_tick_price,
            allow_last_fallback=True,
        )
        if end_fill is not None:
            _close_position(
                realized_exit=float(end_fill),
                exit_ts=last_tick_ts,
                reason="end_of_data",
                event_name="force_close_end_of_data",
                idx=last_tick_idx,
            )

    if telemetry_callback is not None and missing_quote_trigger_count > 0:
        telemetry_callback(
            {
                "event_name": "orderflow_missing_quote_trigger_checks",
                "count": missing_quote_trigger_count,
            }
        )

    wins = sum(1 for t in trades if t.pnl > 0)
    num_trades = len(trades)
    win_rate = (wins / num_trades * 100.0) if num_trades else 0.0
    net_pnl = cash - config.initial_cash
    total_return_pct = (net_pnl / config.initial_cash * 100.0) if config.initial_cash else 0.0

    return BacktestResult(
        initial_cash=config.initial_cash,
        final_equity=cash,
        net_pnl=net_pnl,
        total_return_pct=total_return_pct,
        num_trades=num_trades,
        win_rate_pct=win_rate,
        max_drawdown_pct=max_drawdown * 100.0,
        trades=trades,
    )
