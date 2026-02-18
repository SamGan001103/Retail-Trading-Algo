from _common import load_runtime_and_broker


def main() -> int:
    config, broker = load_runtime_and_broker()
    try:
        positions = broker.open_positions(config.account_id)
        print(f"Open positions: {len(positions)}")
        for position in positions[:50]:
            print(position)
    finally:
        broker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
