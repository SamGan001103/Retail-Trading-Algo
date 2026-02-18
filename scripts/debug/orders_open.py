from _common import load_runtime_and_broker


def main() -> int:
    config, broker = load_runtime_and_broker()
    try:
        orders = broker.open_orders(config.account_id)
        print(f"Open orders: {len(orders)}")
        for order in orders[:50]:
            print(order)
    finally:
        broker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
