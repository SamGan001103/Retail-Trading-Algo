from trading_algo.execution import engine_from_env


def main() -> int:
    _, engine = engine_from_env()
    orders = engine.open_orders()
    print(f"Open orders: {len(orders)}")
    for order in orders[:50]:
        print(order)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
