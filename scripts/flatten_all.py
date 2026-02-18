from trading_algo.execution import engine_from_env


def main() -> int:
    _, engine = engine_from_env()
    cancel_attempted, close_attempted = engine.flatten()
    print(f"Flatten complete. cancel_attempted={cancel_attempted} close_attempted={close_attempted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
