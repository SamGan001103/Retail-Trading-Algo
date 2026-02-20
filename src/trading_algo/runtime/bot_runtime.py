from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any

from trading_algo.broker import broker_from_runtime_config
from trading_algo.config import RuntimeConfig, env_bool, env_float, env_int, get_symbol_profile, load_runtime_config
from trading_algo.core import BUY
from trading_algo.position_management import RiskLimits, enforce_position_limits
from trading_algo.runtime.drawdown_guard import DrawdownGuard
from trading_algo.strategy import MarketBar, OrderFlowState, PositionState, Strategy, StrategyContext
from trading_algo.telemetry import TelemetryRouter


def _filter_by_contract(items: list[dict], contract_id: str) -> list[dict]:
    return [x for x in items if str(x.get("contractId")) == str(contract_id)]


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


def _extract_event_dt(quote: dict[str, Any] | None, trade: dict[str, Any] | None) -> datetime:
    payload = trade or quote or {}
    for key in ("timestamp", "ts", "time", "datetime", "date"):
        raw = payload.get(key)
        if raw is None:
            continue
        if isinstance(raw, (int, float)):
            sec = float(raw)
            if sec > 1e12:
                sec /= 1000.0
            return datetime.fromtimestamp(sec, tz=timezone.utc)
        if isinstance(raw, str):
            value = raw.strip()
            if value.endswith("Z"):
                value = value[:-1] + "+00:00"
            try:
                dt = datetime.fromisoformat(value)
            except ValueError:
                continue
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
    return datetime.now(timezone.utc)


def _extract_price(quote: dict[str, Any] | None, trade: dict[str, Any] | None, last_price: float | None) -> float | None:
    trade_price = _num(trade, "price", "last", "lastPrice", "tradePrice", "close")
    if trade_price is not None:
        return trade_price
    bid = _num(quote, "bid", "bidPrice", "bestBid")
    ask = _num(quote, "ask", "askPrice", "bestAsk")
    if bid is not None and ask is not None:
        return (bid + ask) / 2.0
    if bid is not None:
        return bid
    if ask is not None:
        return ask
    return last_price


def _extract_trade_volume(trade: dict[str, Any] | None) -> float:
    return _num(trade, "size", "qty", "quantity", "volume", "lastSize") or 0.0


def _build_bar(bucket: int, bar_sec: int, o: float, h: float, l: float, c: float, v: float) -> MarketBar:
    end_ts = datetime.fromtimestamp((bucket + 1) * bar_sec, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    return MarketBar(ts=end_ts, open=o, high=h, low=l, close=c, volume=v)


def _position_side(position: dict[str, Any]) -> int | None:
    raw = position.get("side")
    if raw in (0, 1):
        return int(raw)
    direction = str(position.get("direction") or "").strip().lower()
    if direction in {"buy", "long"}:
        return BUY
    if direction in {"sell", "short"}:
        return 1
    return None


def _position_size(position: dict[str, Any]) -> int:
    value = _num(position, "size", "qty", "quantity", "contracts")
    if value is None:
        return 0
    return max(0, int(value))


def _position_entry_price(position: dict[str, Any]) -> float | None:
    return _num(position, "avgPrice", "averagePrice", "entryPrice", "price")


def _tick_size_and_value_from_env(symbol: str) -> tuple[float, float]:
    profile = get_symbol_profile(symbol)
    return (
        env_float("STRAT_TICK_SIZE", profile.tick_size),
        env_float("STRAT_TICK_VALUE", profile.tick_value),
    )


def _drawdown_killswitch_from_env() -> float:
    return env_float("ACCOUNT_MAX_DRAWDOWN_KILLSWITCH", env_float("ACCOUNT_MAX_DRAWDOWN", 2_500.0))


def _strategy_side_name(side: int | None) -> str:
    if side == 0:
        return "buy"
    if side == 1:
        return "sell"
    return "unknown"


def _is_debug(profile: str) -> bool:
    return profile.strip().lower() == "debug"


def _runtime_log(
    profile: str,
    telemetry: TelemetryRouter | None,
    message: str,
    *,
    important: bool = False,
    **fields: Any,
) -> None:
    debug_enabled = _is_debug(profile)
    if telemetry is not None and debug_enabled:
        telemetry.trace(message, **fields)
    elif debug_enabled:
        if fields:
            print(f"{message} {fields}")
        else:
            print(message)
    elif important:
        print(message)


def _drain_candidate_events(
    strategy: Strategy,
    telemetry: TelemetryRouter | None,
    *,
    source: str,
    event_ts: str,
    context_index: int,
) -> None:
    drain = getattr(strategy, "drain_candidate_events", None)
    if not callable(drain):
        return
    try:
        events = drain()
    except Exception:
        return
    if not isinstance(events, list):
        return
    for event in events:
        if not isinstance(event, dict):
            continue
        if telemetry is not None:
            payload: dict[str, Any] = {
                "event_name": "strategy_candidate",
                "source": source,
                "event_ts": event_ts,
                "context_index": context_index,
            }
            payload.update(event)
            telemetry.emit_candidate(payload)


def _run_legacy_loop(
    config: RuntimeConfig,
    broker,
    contract_id: str,
    rt,
    *,
    profile: str,
    telemetry: TelemetryRouter | None,
) -> None:
    traded_once = False
    was_in_position = False
    exit_grace_until = 0.0
    last_price: float | None = None

    limits = RiskLimits(
        max_open_positions=max(1, env_int("STRAT_MAX_OPEN_POSITIONS", 1)),
        max_open_orders_while_flat=max(0, env_int("STRAT_MAX_OPEN_ORDERS_WHILE_FLAT", 0)),
    )
    tick_size, tick_value = _tick_size_and_value_from_env(config.symbol)
    drawdown_guard = DrawdownGuard(
        max_drawdown_abs=_drawdown_killswitch_from_env(),
        tick_size=tick_size,
        tick_value=tick_value,
        enabled=env_bool("STRAT_DRAWDOWN_GUARD_ENABLED", True),
    )
    drawdown_tripped = False

    while True:
        orders_all, positions_all = rt.snapshot()
        orders = _filter_by_contract(orders_all, contract_id)
        positions = _filter_by_contract(positions_all, contract_id)

        in_position = len(positions) > 0
        now = time.time()
        if was_in_position and not in_position:
            exit_grace_until = now + float(config.exit_grace_sec)
        was_in_position = in_position
        in_exit_grace = now < exit_grace_until

        limits_ok, limits_reason = enforce_position_limits(len(positions), len(orders), limits)
        if not limits_ok:
            _runtime_log(
                profile,
                telemetry,
                "position-limits-breach",
                reason=limits_reason,
                contract_id=contract_id,
            )
            cancel_attempted, close_attempted = broker.flatten(config.account_id)
            if telemetry is not None:
                telemetry.emit_execution(
                    {
                        "event_name": "flatten",
                        "source": "legacy_position_limits",
                        "contract_id": contract_id,
                        "cancel_attempted": bool(cancel_attempted),
                        "close_attempted": bool(close_attempted),
                    }
                )
            time.sleep(config.loop_sec)
            continue

        quote = rt.last_quote(contract_id)
        trade = rt.last_trade(contract_id)
        price = _extract_price(quote, trade, last_price)
        if price is not None:
            last_price = price
            if positions:
                p0 = positions[0]
                pos_state = PositionState(
                    in_position=True,
                    side=_position_side(p0),
                    size=_position_size(p0),
                    entry_price=_position_entry_price(p0),
                    bars_in_position=0,
                )
            else:
                pos_state = PositionState(in_position=False)
            dd = drawdown_guard.update(pos_state, price)
            if dd.breached and not drawdown_tripped:
                drawdown_tripped = True
                _runtime_log(
                    profile,
                    telemetry,
                    "drawdown-guard-tripped",
                    drawdown_abs=round(dd.drawdown_abs, 6),
                    max_drawdown_abs=round(drawdown_guard.max_drawdown_abs, 6),
                )
                if telemetry is not None:
                    telemetry.emit_performance(
                        {
                            "event_name": "drawdown_guard_tripped",
                            "drawdown_abs": round(dd.drawdown_abs, 6),
                            "max_drawdown_abs": round(drawdown_guard.max_drawdown_abs, 6),
                        }
                    )
        if drawdown_tripped:
            if in_position and not in_exit_grace:
                cancel_attempted, close_attempted = broker.flatten(config.account_id)
                if telemetry is not None:
                    telemetry.emit_execution(
                        {
                            "event_name": "flatten",
                            "source": "legacy_drawdown_guard",
                            "contract_id": contract_id,
                            "cancel_attempted": bool(cancel_attempted),
                            "close_attempted": bool(close_attempted),
                        }
                    )
            time.sleep(config.loop_sec)
            continue

        if not in_position:
            if config.trade_on_start and (not traded_once) and (not orders) and (not in_exit_grace):
                response = broker.place_market_with_brackets(
                    account_id=config.account_id,
                    contract_id=contract_id,
                    side=config.side if config.side in (0, 1) else BUY,
                    size=config.size,
                    sl_ticks_abs=config.sl_ticks,
                    tp_ticks_abs=config.tp_ticks,
                )
                if telemetry is not None:
                    telemetry.emit_execution(
                        {
                            "event_name": "legacy_entry",
                            "contract_id": contract_id,
                            "side": _strategy_side_name(config.side if config.side in (0, 1) else BUY),
                            "size": int(config.size),
                            "sl_ticks_abs": int(config.sl_ticks),
                            "tp_ticks_abs": int(config.tp_ticks),
                            "success": bool(response.get("success")),
                            "raw_response": response,
                        }
                    )
                if response.get("success"):
                    traded_once = True
                else:
                    _runtime_log(profile, telemetry, "legacy-trade-rejected", response=response)
        else:
            if len(orders) < 2:
                _runtime_log(profile, telemetry, "legacy-missing-brackets", contract_id=contract_id)
                cancel_attempted, close_attempted = broker.flatten(config.account_id)
                if telemetry is not None:
                    telemetry.emit_execution(
                        {
                            "event_name": "flatten",
                            "source": "legacy_missing_brackets",
                            "contract_id": contract_id,
                            "cancel_attempted": bool(cancel_attempted),
                            "close_attempted": bool(close_attempted),
                        }
                    )

        if telemetry is not None and price is not None:
            telemetry.emit_performance(
                {
                    "event_name": "legacy_snapshot",
                    "contract_id": contract_id,
                    "price": round(price, 6),
                    "in_position": bool(in_position),
                    "open_orders": len(orders),
                    "open_positions": len(positions),
                    "drawdown_tripped": bool(drawdown_tripped),
                }
            )

        time.sleep(config.loop_sec)


def _run_strategy_loop(
    config: RuntimeConfig,
    broker,
    contract_id: str,
    rt,
    strategy: Strategy,
    *,
    profile: str,
    telemetry: TelemetryRouter | None,
) -> None:
    bar_sec = max(1, env_int("STRAT_FORWARD_BAR_SEC", max(1, int(config.loop_sec))))
    tick_poll_default = max(0.01, env_float("STRAT_TICK_POLL_SEC", min(0.25, max(0.01, float(config.loop_sec) / 4.0))))
    tick_poll_idle_sec = max(0.01, env_float("STRAT_TICK_POLL_IDLE_SEC", tick_poll_default))
    tick_poll_armed_sec = max(0.005, env_float("STRAT_TICK_POLL_ARMED_SEC", min(0.05, tick_poll_idle_sec)))
    if tick_poll_armed_sec > tick_poll_idle_sec:
        tick_poll_armed_sec = tick_poll_idle_sec
    _runtime_log(
        profile,
        telemetry,
        "strategy-runtime-config",
        strategy_bar_sec=bar_sec,
        strategy_tick_poll_idle_sec=tick_poll_idle_sec,
        strategy_tick_poll_armed_sec=tick_poll_armed_sec,
    )
    limits = RiskLimits(
        max_open_positions=max(1, env_int("STRAT_MAX_OPEN_POSITIONS", 1)),
        max_open_orders_while_flat=max(0, env_int("STRAT_MAX_OPEN_ORDERS_WHILE_FLAT", 0)),
    )
    tick_size, tick_value = _tick_size_and_value_from_env(config.symbol)
    drawdown_guard = DrawdownGuard(
        max_drawdown_abs=_drawdown_killswitch_from_env(),
        tick_size=tick_size,
        tick_value=tick_value,
        enabled=env_bool("STRAT_DRAWDOWN_GUARD_ENABLED", True),
    )
    drawdown_tripped = False

    current_bucket: int | None = None
    bar_open = bar_high = bar_low = bar_close = None
    bar_volume = 0.0
    bar_index = 0
    last_price: float | None = None
    bars_in_position = 0
    was_in_position = False
    exit_grace_until = 0.0
    tick_counter = 0

    if telemetry is not None:
        telemetry.emit_performance(
            {
                "event_name": "runtime_start",
                "contract_id": contract_id,
                "symbol": config.symbol,
                "bar_sec": bar_sec,
                "tick_poll_idle_sec": tick_poll_idle_sec,
                "tick_poll_armed_sec": tick_poll_armed_sec,
                "profile": profile,
            }
        )

    while True:
        orders_all, positions_all = rt.snapshot()
        orders = _filter_by_contract(orders_all, contract_id)
        positions = _filter_by_contract(positions_all, contract_id)

        in_position = len(positions) > 0
        now_ts = time.time()
        if was_in_position and not in_position:
            exit_grace_until = now_ts + float(config.exit_grace_sec)
            bars_in_position = 0
        was_in_position = in_position
        in_exit_grace = now_ts < exit_grace_until

        limits_ok, limits_reason = enforce_position_limits(len(positions), len(orders), limits)
        if not limits_ok:
            _runtime_log(profile, telemetry, "position-limits-breach", reason=limits_reason)
            cancel_attempted, close_attempted = broker.flatten(config.account_id)
            if telemetry is not None:
                telemetry.emit_execution(
                    {
                        "event_name": "flatten",
                        "source": "position_limits",
                        "contract_id": contract_id,
                        "cancel_attempted": bool(cancel_attempted),
                        "close_attempted": bool(close_attempted),
                        "reason": limits_reason,
                    }
                )
            time.sleep(tick_poll_idle_sec)
            continue

        quote = rt.last_quote(contract_id)
        trade = rt.last_trade(contract_id)
        depth = rt.last_depth(contract_id)
        flow = OrderFlowState(quote=quote, trade=trade, depth=depth)
        if hasattr(strategy, "set_orderflow_state"):
            getattr(strategy, "set_orderflow_state")(flow)

        setup_armed = False
        pending_setup_getter = getattr(strategy, "pending_setup", None)
        if callable(pending_setup_getter):
            try:
                setup_armed = pending_setup_getter() is not None
            except Exception:
                setup_armed = False
        poll_sleep_sec = tick_poll_armed_sec if (setup_armed or in_position) else tick_poll_idle_sec

        event_dt = _extract_event_dt(quote, trade)
        price = _extract_price(quote, trade, last_price)
        if price is None:
            time.sleep(poll_sleep_sec)
            continue
        tick_ts = event_dt.isoformat().replace("+00:00", "Z")
        tick_counter += 1

        last_price = price
        bucket = int(event_dt.timestamp()) // bar_sec
        trade_volume = _extract_trade_volume(trade)

        if positions:
            p0 = positions[0]
            pos_state = PositionState(
                in_position=True,
                side=_position_side(p0),
                size=_position_size(p0),
                entry_price=_position_entry_price(p0),
                bars_in_position=bars_in_position,
            )
        else:
            pos_state = PositionState(in_position=False)

        dd = drawdown_guard.update(pos_state, price)
        if dd.breached and not drawdown_tripped:
            drawdown_tripped = True
            _runtime_log(
                profile,
                telemetry,
                "drawdown-guard-tripped",
                drawdown_abs=round(dd.drawdown_abs, 6),
                max_drawdown_abs=round(drawdown_guard.max_drawdown_abs, 6),
            )
            if telemetry is not None:
                telemetry.emit_performance(
                    {
                        "event_name": "drawdown_guard_tripped",
                        "contract_id": contract_id,
                        "drawdown_abs": round(dd.drawdown_abs, 6),
                        "max_drawdown_abs": round(drawdown_guard.max_drawdown_abs, 6),
                    }
                )
        if drawdown_tripped:
            if in_position and not in_exit_grace:
                cancel_attempted, close_attempted = broker.flatten(config.account_id)
                if telemetry is not None:
                    telemetry.emit_execution(
                        {
                            "event_name": "flatten",
                            "source": "drawdown_guard",
                            "contract_id": contract_id,
                            "cancel_attempted": bool(cancel_attempted),
                            "close_attempted": bool(close_attempted),
                            "reason": "drawdown_guard_tripped",
                        }
                    )
            time.sleep(poll_sleep_sec)
            continue

        tick_handler = getattr(strategy, "on_tick", None)
        should_run_tick_handler = callable(tick_handler) and (setup_armed or in_position)
        if should_run_tick_handler:
            tick_decision = tick_handler(
                tick_ts,
                price,
                StrategyContext(index=bar_index, total_bars=bar_index + 1),
                pos_state,
            )
            if tick_decision.should_exit and in_position and not in_exit_grace:
                cancel_attempted, close_attempted = broker.flatten(config.account_id)
                if telemetry is not None:
                    telemetry.emit_execution(
                        {
                            "event_name": "strategy_tick_exit_flatten",
                            "contract_id": contract_id,
                            "reason": tick_decision.reason,
                            "cancel_attempted": bool(cancel_attempted),
                            "close_attempted": bool(close_attempted),
                        }
                    )
            elif tick_decision.should_enter and (not in_position) and (not orders) and (not in_exit_grace):
                sl_ticks = int(tick_decision.sl_ticks_abs) if tick_decision.sl_ticks_abs else int(config.sl_ticks)
                tp_ticks = int(tick_decision.tp_ticks_abs) if tick_decision.tp_ticks_abs else int(config.tp_ticks)
                response = broker.place_market_with_brackets(
                    account_id=config.account_id,
                    contract_id=contract_id,
                    side=tick_decision.side if tick_decision.side in (0, 1) else BUY,
                    size=max(1, int(tick_decision.size)),
                    sl_ticks_abs=max(1, sl_ticks),
                    tp_ticks_abs=max(1, tp_ticks),
                )
                if telemetry is not None:
                    telemetry.emit_execution(
                        {
                            "event_name": "strategy_tick_entry",
                            "contract_id": contract_id,
                            "reason": tick_decision.reason,
                            "side": _strategy_side_name(tick_decision.side if tick_decision.side in (0, 1) else BUY),
                            "size": max(1, int(tick_decision.size)),
                            "sl_ticks_abs": max(1, sl_ticks),
                            "tp_ticks_abs": max(1, tp_ticks),
                            "success": bool(response.get("success")),
                            "raw_response": response,
                        }
                    )
                if not response.get("success"):
                    _runtime_log(profile, telemetry, "strategy-tick-entry-rejected", response=response)
            _drain_candidate_events(
                strategy,
                telemetry,
                source="tick",
                event_ts=tick_ts,
                context_index=bar_index,
            )

        if telemetry is not None and (_is_debug(profile) and (tick_counter % telemetry.debug_tick_trace_every_n() == 0)):
            telemetry.emit_performance(
                {
                    "event_name": "tick_trace",
                    "tick_ts": tick_ts,
                    "bar_index": bar_index,
                    "price": round(price, 6),
                    "setup_armed": bool(setup_armed),
                    "in_position": bool(in_position),
                    "open_orders": len(orders),
                    "open_positions": len(positions),
                }
            )

        if current_bucket is None:
            current_bucket = bucket
            bar_open = bar_high = bar_low = bar_close = price
            bar_volume = trade_volume
            time.sleep(poll_sleep_sec)
            continue

        if bucket != current_bucket:
            assert bar_open is not None and bar_high is not None and bar_low is not None and bar_close is not None
            completed = _build_bar(current_bucket, bar_sec, bar_open, bar_high, bar_low, bar_close, bar_volume)
            decision = strategy.on_bar(completed, StrategyContext(index=bar_index, total_bars=bar_index + 1), pos_state)
            _drain_candidate_events(
                strategy,
                telemetry,
                source="bar",
                event_ts=completed.ts,
                context_index=bar_index,
            )
            if decision.should_exit and in_position and not in_exit_grace:
                cancel_attempted, close_attempted = broker.flatten(config.account_id)
                if telemetry is not None:
                    telemetry.emit_execution(
                        {
                            "event_name": "strategy_bar_exit_flatten",
                            "contract_id": contract_id,
                            "reason": decision.reason,
                            "cancel_attempted": bool(cancel_attempted),
                            "close_attempted": bool(close_attempted),
                        }
                    )
            elif decision.should_enter and (not in_position) and (not orders) and (not in_exit_grace):
                sl_ticks = int(decision.sl_ticks_abs) if decision.sl_ticks_abs else int(config.sl_ticks)
                tp_ticks = int(decision.tp_ticks_abs) if decision.tp_ticks_abs else int(config.tp_ticks)
                response = broker.place_market_with_brackets(
                    account_id=config.account_id,
                    contract_id=contract_id,
                    side=decision.side if decision.side in (0, 1) else BUY,
                    size=max(1, int(decision.size)),
                    sl_ticks_abs=max(1, sl_ticks),
                    tp_ticks_abs=max(1, tp_ticks),
                )
                if telemetry is not None:
                    telemetry.emit_execution(
                        {
                            "event_name": "strategy_bar_entry",
                            "contract_id": contract_id,
                            "reason": decision.reason,
                            "side": _strategy_side_name(decision.side if decision.side in (0, 1) else BUY),
                            "size": max(1, int(decision.size)),
                            "sl_ticks_abs": max(1, sl_ticks),
                            "tp_ticks_abs": max(1, tp_ticks),
                            "success": bool(response.get("success")),
                            "raw_response": response,
                        }
                    )
                if not response.get("success"):
                    _runtime_log(profile, telemetry, "strategy-bar-entry-rejected", response=response)

            if in_position:
                bars_in_position += 1
            else:
                bars_in_position = 0
            bar_index += 1

            if telemetry is not None:
                telemetry.emit_performance(
                    {
                        "event_name": "bar_snapshot",
                        "bar_ts": completed.ts,
                        "bar_index": bar_index,
                        "bar_open": round(completed.open, 6),
                        "bar_high": round(completed.high, 6),
                        "bar_low": round(completed.low, 6),
                        "bar_close": round(completed.close, 6),
                        "bar_volume": round(completed.volume, 6),
                        "decision_reason": decision.reason,
                        "decision_should_enter": bool(decision.should_enter),
                        "decision_should_exit": bool(decision.should_exit),
                        "in_position": bool(in_position),
                        "open_orders": len(orders),
                        "open_positions": len(positions),
                        "drawdown_abs": round(dd.drawdown_abs, 6),
                        "drawdown_peak_pnl": round(dd.peak_pnl, 6),
                        "drawdown_equity_pnl": round(dd.equity_pnl, 6),
                    }
                )

            current_bucket = bucket
            bar_open = bar_high = bar_low = bar_close = price
            bar_volume = trade_volume
        else:
            assert bar_high is not None and bar_low is not None
            bar_high = max(bar_high, price)
            bar_low = min(bar_low, price)
            bar_close = price
            bar_volume += trade_volume

        if in_position and len(orders) < 2:
            _runtime_log(profile, telemetry, "missing-brackets-while-in-position")
            cancel_attempted, close_attempted = broker.flatten(config.account_id)
            if telemetry is not None:
                telemetry.emit_execution(
                    {
                        "event_name": "flatten",
                        "source": "missing_brackets",
                        "contract_id": contract_id,
                        "cancel_attempted": bool(cancel_attempted),
                        "close_attempted": bool(close_attempted),
                    }
                )

        time.sleep(poll_sleep_sec)


def run(
    config: RuntimeConfig,
    strategy: Strategy | None = None,
    *,
    profile: str = "normal",
    telemetry: TelemetryRouter | None = None,
) -> None:
    profile = profile.strip().lower() if profile else "normal"
    broker = broker_from_runtime_config(config)
    rt = None
    telemetry_router = telemetry
    if telemetry_router is None:
        strategy_name = strategy.__class__.__name__ if strategy is not None else "legacy"
        telemetry_router = TelemetryRouter.from_env(profile=profile, mode="forward", strategy=strategy_name)

    try:
        contract_id = broker.resolve_contract_id(config.symbol, config.live)
        _runtime_log(
            profile,
            telemetry_router,
            f"contract_id={contract_id} symbol={config.symbol} live={config.live}",
            important=True,
        )

        rt = broker.create_stream(account_id=config.account_id)
        rt.start(contract_id=contract_id)

        if config.flatten_on_start:
            cancel_attempted, close_attempted = broker.flatten(config.account_id)
            if telemetry_router is not None:
                telemetry_router.emit_execution(
                    {
                        "event_name": "flatten_on_start",
                        "contract_id": contract_id,
                        "cancel_attempted": bool(cancel_attempted),
                        "close_attempted": bool(close_attempted),
                    }
                )
            _runtime_log(
                profile,
                telemetry_router,
                f"flatten-on-start cancel_attempted={cancel_attempted} close_attempted={close_attempted}",
                important=_is_debug(profile),
            )

        if strategy is None:
            _run_legacy_loop(
                config,
                broker,
                contract_id,
                rt,
                profile=profile,
                telemetry=telemetry_router,
            )
        else:
            _runtime_log(
                profile,
                telemetry_router,
                f"strategy_runtime={strategy.__class__.__name__}",
                important=True,
            )
            if os.getenv("STRAT_REQUIRE_ORDERFLOW", "").strip().lower() in {"1", "true", "yes", "y", "on"}:
                _runtime_log(profile, telemetry_router, "orderflow_gate=enabled", important=True)
            _run_strategy_loop(
                config,
                broker,
                contract_id,
                rt,
                strategy,
                profile=profile,
                telemetry=telemetry_router,
            )
    except KeyboardInterrupt:
        _runtime_log(profile, telemetry_router, "Stopping bot.", important=True)
    finally:
        if rt is not None:
            rt.stop()
        broker.close()


def main() -> None:
    run(load_runtime_config())


if __name__ == "__main__":
    main()
