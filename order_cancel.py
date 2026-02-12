# order_cancel.py
# Usage:
#   ACCOUNT_ID=18672085 ORDER_ID=2441409238 python order_cancel.py

import os
import sys
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
    order_id   = int(must_env("ORDER_ID"))

    token = login_key(base_url, username, api_key)

    data = post_json(
        base_url=base_url,
        token=token,
        path="/api/Order/cancel",
        payload={"accountId": account_id, "orderId": order_id},
        label="ORDER_CANCEL",
    )

    if not data.get("success"):
        raise RuntimeError(data)

    print("✅ Cancel request accepted:", data)
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
