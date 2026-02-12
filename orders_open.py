# orders_open.py
# Usage:
#   ACCOUNT_ID=18672085 python orders_open.py

import os
import sys
import json
from dotenv import load_dotenv
from projectx_api import login_key, post_json

load_dotenv()

def must_env(name: str) -> str:
    v = (os.getenv(name) or "").strip()
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v

def main() -> int:
    base_url  = must_env("PROJECTX_BASE_URL").rstrip("/")
    username  = must_env("PROJECTX_USERNAME")
    api_key   = must_env("PROJECTX_API_KEY")
    account_id = int(must_env("ACCOUNT_ID"))

    token = login_key(base_url, username, api_key)

    data = post_json(
        base_url=base_url,
        token=token,
        path="/api/Order/searchOpen",
        payload={"accountId": account_id},
        label="ORDER_SEARCH_OPEN",
    )

    if not data.get("success"):
        raise RuntimeError(data)

    orders = data.get("orders") or []
    print(f"✅ Open orders: {len(orders)}")
    for o in orders[:50]:
        print(json.dumps({
            "id": o.get("id"),
            "contractId": o.get("contractId"),
            "symbolId": o.get("symbolId"),
            "status": o.get("status"),
            "type": o.get("type"),
            "side": o.get("side"),
            "size": o.get("size"),
            "limitPrice": o.get("limitPrice"),
            "stopPrice": o.get("stopPrice"),
            "creationTimestamp": o.get("creationTimestamp"),
            "updateTimestamp": o.get("updateTimestamp"),
        }, indent=2))

    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
