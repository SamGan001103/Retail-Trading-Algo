from _common import load_runtime_and_broker


def main() -> int:
    config, broker = load_runtime_and_broker()
    try:
        contracts = broker.search_contracts(config.symbol, config.live)
        print(f"Contracts returned: {len(contracts)}")
        for contract in contracts[:20]:
            print(f"id={contract.get('id')} name={contract.get('name')} desc={contract.get('description')}")
    finally:
        broker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
