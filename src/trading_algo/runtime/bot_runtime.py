from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any

from trading_algo.broker import broker_from_runtime_config
from trading_algo.config import RuntimeConfig, env_float, env_int, load_runtime_config
from trading_algo.core import BUY
from trading_algo.strategy import MarketBar, OrderFlowState, PositionState, Strategy, StrategyContext


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


def _run_legacy_loop(config: RuntimeConfig, broker, contract_id: str, rt) -> None:
    traded_once = False
    was_in_position = False
    exit_grace_until = 0.0

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
                print("place_market_with_brackets:", response)
                if response.get("success"):
                    traded_once = True
                else:
                    print("Trade rejected:", response)
        else:
            if len(orders) < 2:
                print("Missing bracket(s) while in position, flattening.")
                cancel_attempted, close_attempted = broker.flatten(config.account_id)
                print(f"Flatten: cancel_attempted={cancel_attempted} close_attempted={close_attempted}")

        time.sleep(config.loop_sec)


def _run_strategy_loop(config: RuntimeConfig, broker, contract_id: str, rt, strategy: Strategy) -> None:
    bar_sec = max(1, env_int("STRAT_FORWARD_BAR_SEC", max(1, int(config.loop_sec))))
    tick_poll_default = max(0.01, env_float("STRAT_TICK_POLL_SEC", min(0.25, max(0.01, float(config.loop_sec) / 4.0))))
    tick_poll_idle_sec = max(0.01, env_float("STRAT_TICK_POLL_IDLE_SEC", tick_poll_default))
    tick_poll_armed_sec = max(0.005, env_float("STRAT_TICK_POLL_ARMED_SEC", min(0.05, tick_poll_idle_sec)))
    if tick_poll_armed_sec > tick_poll_idle_sec:
        tick_poll_armed_sec = tick_poll_idle_sec
    print(f"strategy_bar_sec={bar_sec}")
    print(f"strategy_tick_poll_idle_sec={tick_poll_idle_sec}")
    print(f"strategy_tick_poll_armed_sec={tick_poll_armed_sec}")

    current_bucket: int | None = None
    bar_open = bar_high = bar_low = bar_close = None
    bar_volume = 0.0
    bar_index = 0
    last_price: float | None = None
    bars_in_position = 0
    was_in_position = False
    exit_grace_until = 0.0

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
                print(f"strategy-tick-exit flatten: cancel_attempted={cancel_attempted} close_attempted={close_attempted}")
            elif tick_decision.should_enter and (not in_position) and (not orders) and (not in_exit_grace):
                response = broker.place_market_with_brackets(
                    account_id=config.account_id,
                    contract_id=contract_id,
                    side=tick_decision.side if tick_decision.side in (0, 1) else BUY,
                    size=max(1, int(tick_decision.size)),
                    sl_ticks_abs=config.sl_ticks,
                    tp_ticks_abs=config.tp_ticks,
                )
                print(f"strategy-tick-entry {tick_decision.reason}:", response)
                if not response.get("success"):
                    print("Trade rejected:", response)

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
            if decision.should_exit and in_position and not in_exit_grace:
                cancel_attempted, close_attempted = broker.flatten(config.account_id)
                print(f"strategy-exit flatten: cancel_attempted={cancel_attempted} close_attempted={close_attempted}")
            elif decision.should_enter and (not in_position) and (not orders) and (not in_exit_grace):
                response = broker.place_market_with_brackets(
                    account_id=config.account_id,
                    contract_id=contract_id,
                    side=decision.side if decision.side in (0, 1) else BUY,
                    size=max(1, int(decision.size)),
                    sl_ticks_abs=config.sl_ticks,
                    tp_ticks_abs=config.tp_ticks,
                )
                print(f"strategy-entry {decision.reason}:", response)
                if not response.get("success"):
                    print("Trade rejected:", response)

            if in_position:
                bars_in_position += 1
            else:
                bars_in_position = 0
            bar_index += 1

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
            print("Missing bracket(s) while in position, flattening.")
            cancel_attempted, close_attempted = broker.flatten(config.account_id)
            print(f"Flatten: cancel_attempted={cancel_attempted} close_attempted={close_attempted}")

        time.sleep(poll_sleep_sec)


def run(config: RuntimeConfig, strategy: Strategy | None = None) -> None:
    broker = broker_from_runtime_config(config)
    rt = None

    try:
        contract_id = broker.resolve_contract_id(config.symbol, config.live)
        print(f"contract_id={contract_id} symbol={config.symbol} live={config.live}")

        rt = broker.create_stream(account_id=config.account_id)
        rt.start(contract_id=contract_id)

        if config.flatten_on_start:
            cancel_attempted, close_attempted = broker.flatten(config.account_id)
            print(f"Flatten on start: cancel_attempted={cancel_attempted} close_attempted={close_attempted}")

        if strategy is None:
            _run_legacy_loop(config, broker, contract_id, rt)
        else:
            print(f"strategy_runtime={strategy.__class__.__name__}")
            if os.getenv("STRAT_REQUIRE_ORDERFLOW", "").strip().lower() in {"1", "true", "yes", "y", "on"}:
                print("orderflow_gate=enabled (uses broker stream depth via adapter)")
            _run_strategy_loop(config, broker, contract_id, rt, strategy)
    except KeyboardInterrupt:
        print("Stopping bot.")
    finally:
        if rt is not None:
            rt.stop()
        broker.close()


def main() -> None:
    run(load_runtime_config())


if __name__ == "__main__":
    main()
