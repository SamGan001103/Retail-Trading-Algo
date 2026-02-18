from _common import load_runtime_and_broker


def main() -> int:
    config, broker = load_runtime_and_broker()
    try:
        accounts = broker.list_accounts()
        target = next((a for a in accounts if int(a.get("id")) == config.account_id), None)
        if not target:
            raise RuntimeError(f"ACCOUNT_ID {config.account_id} not found in active accounts")
        print("Selected account:")
        print(f"id={target.get('id')} name={target.get('name')} canTrade={target.get('canTrade')}")
    finally:
        broker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
