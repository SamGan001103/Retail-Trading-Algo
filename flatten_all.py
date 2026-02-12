# flatten_all.py
# Usage:
#   ACCOUNT_ID=18672085 python flatten_all.py

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

    token = login_key(base_url, username, api_key)

    # 1) Cancel open orders
    open_orders = post_json(
        base_url, token, "/api/Order/searchOpen",
        {"accountId": account_id},
        "ORDER_SEARCH_OPEN"
    )
    if not open_orders.get("success"):
        raise RuntimeError(open_orders)

    orders = open_orders.get("orders") or []
    print(f"Open orders: {len(orders)}")
    for o in orders:
        oid = o.get("id")
        if oid is None:
            continue
        resp = post_json(
            base_url, token, "/api/Order/cancel",
            {"accountId": account_id, "orderId": int(oid)},
            f"ORDER_CANCEL({oid})"
        )
        if not resp.get("success"):
            print("⚠️ cancel failed:", resp)

    # 2) Close open positions
    open_pos = post_json(
        base_url, token, "/api/Position/searchOpen",
        {"accountId": account_id},
        "POSITION_SEARCH_OPEN"
    )
    if not open_pos.get("success"):
        raise RuntimeError(open_pos)

    positions = open_pos.get("positions") or []
    print(f"Open positions: {len(positions)}")
    for p in positions:
        cid = p.get("contractId")
        if not cid:
            continue
        resp = post_json(
            base_url, token, "/api/Position/closeContract",
            {"accountId": account_id, "contractId": cid},
            f"POSITION_CLOSE_CONTRACT({cid})"
        )
        if not resp.get("success"):
            print("⚠️ close failed:", resp)

    print("✅ Flatten complete (cancel orders → close positions).")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
