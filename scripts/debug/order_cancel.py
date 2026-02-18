from trading_algo.config import must_env
from trading_algo.execution import engine_from_env


def main() -> int:
    _, engine = engine_from_env()
    response = engine.cancel_order(int(must_env("ORDER_ID")))
    print(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
