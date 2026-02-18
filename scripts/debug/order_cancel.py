from _common import load_runtime_and_broker
from trading_algo.config import must_env


def main() -> int:
    config, broker = load_runtime_and_broker()
    try:
        response = broker.cancel_order(config.account_id, int(must_env("ORDER_ID")))
        print(response)
    finally:
        broker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
