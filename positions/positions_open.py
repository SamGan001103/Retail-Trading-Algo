# positions_open.py
# Usage:
#   ACCOUNT_ID=18672085 python positions_open.py

import os
import sys
import json
from dotenv import load_dotenv

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, 'core'))

from projectx_api import login_key, post_json  # type: ignore

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
        path="/api/Position/searchOpen",
        payload={"accountId": account_id},
        label="POSITION_SEARCH_OPEN",
    )

    if not data.get("success"):
        raise RuntimeError(data)

    positions = data.get("positions") or []
    print(f"✅ Open positions: {len(positions)}")
    for p in positions[:50]:
        print(json.dumps({
            "contractId": p.get("contractId"),
            "symbolId": p.get("symbolId"),
            "netPos": p.get("netPos"),
            "avgPrice": p.get("avgPrice"),
            "unrealizedPnl": p.get("unrealizedPnl"),
        }, indent=2))

    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
