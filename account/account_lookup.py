# account_lookup.py
import os
import sys
import json
from dotenv import load_dotenv

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, 'core'))

from projectx_api import login_key, post_json  # type: ignore


load_dotenv()

def must_get(name: str) -> str:
    v = (os.getenv(name) or "").strip().strip('"').strip("'")
    if not v:
        raise RuntimeError(f"Missing {name} in .env")
    return v

def main() -> int:
    base_url = must_get("PROJECTX_BASE_URL").rstrip("/")
    username = must_get("PROJECTX_USERNAME")
    api_key  = must_get("PROJECTX_API_KEY")

    token = login_key(base_url, username, api_key)
    print("✅ Login OK")

    data = post_json(
        base_url, token,
        "/api/Account/search",
        {"onlyActiveAccounts": True},
        "ACCOUNT_SEARCH"
    )

    if not data.get("success"):
        raise RuntimeError(f"Account search failed: {data}")

    accounts = data.get("accounts") or []
    print(f"✅ Returned {len(accounts)} account(s)")

    # Print a compact view (defensive field access)
    for a in accounts:
        print(json.dumps({
            "id": a.get("id"),
            "name": a.get("name"),
            "status": a.get("status"),
            "isActive": a.get("isActive"),
            "accountType": a.get("accountType"),
        }, indent=2))

    if accounts:
        print("\n🎯 Next step:")
        print("Pick one account 'id' and export it for order tests, e.g.:")
        print("  ACCOUNT_ID=123456 python order_smoke_test.py")

    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
