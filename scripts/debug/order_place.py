from _common import load_runtime_and_broker


def main() -> int:
    config, broker = load_runtime_and_broker()
    try:
        contract_id = broker.resolve_contract_id(config.symbol, config.live)
        response = broker.place_market_with_brackets(
            account_id=config.account_id,
            contract_id=contract_id,
            side=config.side,
            size=config.size,
            sl_ticks_abs=config.sl_ticks,
            tp_ticks_abs=config.tp_ticks,
        )
        print(response)
    finally:
        broker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
