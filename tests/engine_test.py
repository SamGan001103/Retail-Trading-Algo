# engine_test.py
# Usage:
#   python engine_test.py

import os
import sys
from dotenv import load_dotenv

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, 'core'))
sys.path.insert(0, os.path.join(root_dir, 'market'))

from execution_engine import ExecutionEngine  # type: ignore
from market_lookup import contract_search, must_get, login  # type: ignore

load_dotenv()

def main() -> None:
    base_url = must_get("PROJECTX_BASE_URL").rstrip("/")
    username = must_get("PROJECTX_USERNAME")
    api_key  = must_get("PROJECTX_API_KEY")
    account_id = int((os.getenv("ACCOUNT_ID") or "").strip())

    eng = ExecutionEngine(base_url, username, api_key, account_id)

    ok, reason = eng.can_enter_trade()
    print("can_enter_trade:", ok, "-", reason)

    orders = eng.open_orders()
    positions = eng.open_positions()
    print("open_orders:", len(orders))
    print("open_positions:", len(positions))

if __name__ == "__main__":
    main()
