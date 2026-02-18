from __future__ import annotations

import time

from trading_algo.api import ProjectXClient, resolve_contract_id
from trading_algo.config import RuntimeConfig, load_runtime_config
from trading_algo.execution import BUY, ExecutionEngine
from trading_algo.runtime.realtime_client import RealtimeClient


def _filter_by_contract(items: list[dict], contract_id: str) -> list[dict]:
    return [x for x in items if str(x.get("contractId")) == str(contract_id)]


def run(config: RuntimeConfig) -> None:
    client = ProjectXClient(config.base_url, config.username, config.api_key)
    engine = ExecutionEngine(client, config.account_id)
    contract_id = resolve_contract_id(client, config.symbol, config.live)
    print(f"contract_id={contract_id} symbol={config.symbol} live={config.live}")

    rt = RealtimeClient(
        client=client,
        account_id=config.account_id,
        user_hub_url=config.user_hub_url,
        market_hub_url=config.market_hub_url,
    )
    rt.start(contract_id=contract_id)

    if config.flatten_on_start:
        cancel_attempted, close_attempted = engine.flatten()
        print(f"Flatten on start: cancel_attempted={cancel_attempted} close_attempted={close_attempted}")

    traded_once = False
    was_in_position = False
    exit_grace_until = 0.0

    try:
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
                    response = engine.place_market_with_brackets(
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
                    cancel_attempted, close_attempted = engine.flatten()
                    print(f"Flatten: cancel_attempted={cancel_attempted} close_attempted={close_attempted}")

            time.sleep(config.loop_sec)
    except KeyboardInterrupt:
        print("Stopping bot.")
    finally:
        rt.stop()


def main() -> None:
    run(load_runtime_config())


if __name__ == "__main__":
    main()
