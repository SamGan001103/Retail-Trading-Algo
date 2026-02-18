from trading_algo.execution import engine_from_env


def main() -> int:
    _, engine = engine_from_env()
    positions = engine.open_positions()
    print(f"Open positions: {len(positions)}")
    for position in positions[:50]:
        print(position)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
