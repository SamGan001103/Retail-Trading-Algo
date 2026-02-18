import os

from trading_algo.api import client_from_env, search_contracts
from trading_algo.config import env_bool


def main() -> int:
    symbol = (os.getenv("SYMBOL") or "MNQ").strip().upper()
    live = env_bool("LIVE", False)
    client = client_from_env()
    contracts = search_contracts(client, symbol, live)
    print(f"Contracts returned: {len(contracts)}")
    for contract in contracts[:20]:
        print(f"id={contract.get('id')} name={contract.get('name')} desc={contract.get('description')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
