import json

from trading_algo.api import client_from_env


def main() -> int:
    client = client_from_env()
    data = client.post_json("/api/Account/search", {"onlyActiveAccounts": True}, "ACCOUNT_SEARCH")
    if not data.get("success"):
        raise RuntimeError(f"Account search failed: {data}")
    accounts = data.get("accounts") or []
    print(f"Returned {len(accounts)} account(s)")
    for account in accounts:
        print(
            json.dumps(
                {
                    "id": account.get("id"),
                    "name": account.get("name"),
                    "status": account.get("status"),
                    "isActive": account.get("isActive"),
                    "accountType": account.get("accountType"),
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
