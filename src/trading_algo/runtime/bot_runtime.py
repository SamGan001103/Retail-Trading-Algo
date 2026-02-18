from __future__ import annotations

import time

from trading_algo.broker import broker_from_runtime_config
from trading_algo.config import RuntimeConfig, load_runtime_config
from trading_algo.core import BUY


def _filter_by_contract(items: list[dict], contract_id: str) -> list[dict]:
    return [x for x in items if str(x.get("contractId")) == str(contract_id)]


def run(config: RuntimeConfig) -> None:
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
