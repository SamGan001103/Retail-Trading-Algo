from trading_algo.api import client_from_env
from trading_algo.config import must_env


def main() -> int:
    account_id = int(must_env("ACCOUNT_ID"))
    client = client_from_env()
    data = client.post_json("/api/Account/search", {"onlyActiveAccounts": True}, "ACCOUNT_SEARCH")
    if not data.get("success"):
        raise RuntimeError(f"Account search failed: {data}")
    accounts = data.get("accounts") or []
    target = next((a for a in accounts if int(a.get("id")) == account_id), None)
    if not target:
        raise RuntimeError(f"ACCOUNT_ID {account_id} not found in active accounts")
    print("Selected account:")
    print(f"id={target.get('id')} name={target.get('name')} canTrade={target.get('canTrade')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
