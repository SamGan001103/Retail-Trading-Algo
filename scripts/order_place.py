import os

from trading_algo.api import resolve_contract_id
from trading_algo.config import env_bool, env_int
from trading_algo.execution import engine_from_env


def main() -> int:
    client, engine = engine_from_env()
    symbol = (os.getenv("SYMBOL") or "MNQ").strip().upper()
    live = env_bool("LIVE", False)
    side = env_int("SIDE", 0)
    size = env_int("SIZE", 1)
    sl_ticks = env_int("SL_TICKS", 40)
    tp_ticks = env_int("TP_TICKS", 80)

    contract_id = resolve_contract_id(client, symbol, live)
    response = engine.place_market_with_brackets(
        contract_id=contract_id,
        side=side,
        size=size,
        sl_ticks_abs=sl_ticks,
        tp_ticks_abs=tp_ticks,
    )
    print(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
