# order_place.py
# Usage:
#   ACCOUNT_ID=18672085 SYMBOL=MNQ SIDE=0 SIZE=1 SL_TICKS=40 TP_TICKS=80 python order_place.py
#
# SIDE: 0=BUY (LONG), 1=SELL (SHORT)
# Bracket tick sign convention (enforced by ProjectX):
#   LONG  (SIDE=0): SL must be NEGATIVE, TP must be POSITIVE
#   SHORT (SIDE=1): SL must be POSITIVE, TP must be NEGATIVE

import os
import sys
import json
import requests
from dotenv import load_dotenv

load_dotenv()

def must_get(name: str) -> str:
    v = (os.getenv(name) or "").strip().strip('"').strip("'")
    if not v:
        raise RuntimeError(f"Missing {name} in .env")
    return v

def env_int(name: str, default: int) -> int:
    v = (os.getenv(name) or "").strip()
    return int(v) if v else default

BASE_URL = must_get("PROJECTX_BASE_URL").rstrip("/")

def debug_response(resp: requests.Response, label: str) -> None:
    ct = (resp.headers.get("content-type") or "").lower()
    text = resp.text or ""
    print(f"[DEBUG] {label} status={resp.status_code} content-type={ct} len={len(text)}")
    if text:
        print(f"[DEBUG] {label} body (first 800): {text[:800]!r}")

def login(username: str, api_key: str) -> str:
    url = f"{BASE_URL}/api/Auth/loginKey"
    r = requests.post(
        url,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        json={"userName": username, "apiKey": api_key},
        timeout=30,
    )
    debug_response(r, "LOGIN")

    try:
        data = r.json()
    except Exception:
        raise RuntimeError(f"Login returned non-JSON. HTTP {r.status_code}. Body: {r.text!r}")

    if not data.get("success") or not data.get("token"):
        raise RuntimeError(f"Login failed: HTTP {r.status_code} {data}")

    return data["token"]

def post_json(token: str, path: str, payload: dict, label: str) -> dict:
    url = f"{BASE_URL}{path}"
    print(f"[DEBUG] Sending payload to {path}: {payload}")

    r = requests.post(
        url,
        headers={
            "accept": "text/plain",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        json=payload,
        timeout=30,
    )
    debug_response(r, label)

    try:
        return r.json()
    except Exception:
        raise RuntimeError(f"{label} returned non-JSON. HTTP {r.status_code}. Body: {r.text!r}")

def contract_search(token: str, search_text: str, live: bool) -> dict:
    # Doc-correct flat shape
    payload = {"live": live, "searchText": search_text}
    return post_json(token, "/api/Contract/search", payload, f"CONTRACT_SEARCH(live={live})")

def pick_contract_id(data: dict) -> str:
    if not isinstance(data, dict) or not data.get("success"):
        raise RuntimeError(f"Contract search failed: {data}")
    contracts = data.get("contracts") or []
    if not contracts:
        raise RuntimeError(f"No contracts returned: {data}")
    return contracts[0]["id"]

def signed_brackets(side: int, sl_ticks_abs: int, tp_ticks_abs: int) -> tuple[int, int]:
    sl = abs(int(sl_ticks_abs))
    tp = abs(int(tp_ticks_abs))
    if side == 0:      # LONG (BUY)
        return (-sl, +tp)
    elif side == 1:    # SHORT (SELL)
        return (+sl, -tp)
    else:
        raise RuntimeError(f"Invalid SIDE={side} (expected 0 or 1)")

def main() -> int:
    # Required runtime env
    account_id = (os.getenv("ACCOUNT_ID") or "").strip()
    if not account_id:
        raise RuntimeError("Missing ACCOUNT_ID (e.g. ACCOUNT_ID=18672085)")

    symbol = (os.getenv("SYMBOL") or "MNQ").strip().upper()
    live   = (os.getenv("LIVE") or "false").strip().lower() == "true"  # keep SIM by default

    side = env_int("SIDE", 0)     # 0=BUY (LONG), 1=SELL (SHORT)
    size = env_int("SIZE", 1)

    sl_abs = env_int("SL_TICKS", 40)
    tp_abs = env_int("TP_TICKS", 80)
    sl_ticks, tp_ticks = signed_brackets(side, sl_abs, tp_abs)

    print(f"BASE_URL  = {BASE_URL}")
    print(f"ACCOUNT_ID= {account_id}")
    print(f"SYMBOL    = {symbol}")
    print(f"LIVE      = {live}")
    print(f"SIDE      = {side}  (0=BUY/LONG, 1=SELL/SHORT)")
    print(f"SIZE      = {size}")
    print(f"[DEBUG] Signed brackets: SL={sl_ticks} TP={tp_ticks}")

    username = must_get("PROJECTX_USERNAME")
    api_key  = must_get("PROJECTX_API_KEY")

    # Login
    token = login(username, api_key)
    print("✅ Login OK")

    # Resolve contractId
    cs = contract_search(token, symbol, live=live)
    contract_id = pick_contract_id(cs)
    print(f"✅ Using contractId={contract_id}")

    # Place order + brackets (SIGNED ticks!)
    payload = {
        "accountId": int(account_id),
        "contractId": contract_id,
        "type": 2,          # Market
        "side": int(side),  # 0 buy, 1 sell
        "size": int(size),
        "stopLossBracket": {
            "ticks": int(sl_ticks),   # IMPORTANT: signed
            "type": 4,                # Stop
        },
        "takeProfitBracket": {
            "ticks": int(tp_ticks),   # IMPORTANT: signed
            "type": 1,                # Limit
        },
    }

    print("[DEBUG] Final ORDER payload:")
    print(json.dumps(payload, indent=2))

    data = post_json(token, "/api/Order/place", payload, "ORDER_PLACE")
    if not data.get("success"):
        raise RuntimeError(f"Order place failed: {data}")

    print("✅ Order placed:")
    print(json.dumps(data, indent=2))
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
