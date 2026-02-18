from _common import load_runtime_and_broker


def main() -> int:
    config, broker = load_runtime_and_broker()
    try:
        cancel_attempted, close_attempted = broker.flatten(config.account_id)
        print(f"Flatten complete. cancel_attempted={cancel_attempted} close_attempted={close_attempted}")
    finally:
        broker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
