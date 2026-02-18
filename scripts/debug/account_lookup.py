import json

from _common import load_runtime_and_broker


def main() -> int:
    _, broker = load_runtime_and_broker()
    try:
        accounts = broker.list_accounts()
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
    finally:
        broker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
