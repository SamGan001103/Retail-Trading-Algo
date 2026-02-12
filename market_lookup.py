# market_lookup.py
# Usage:
#   python market_lookup.py
#   SYMBOL=MNQ python market_lookup.py
#
# Requires .env:
#   PROJECTX_BASE_URL=https://api.topstepx.com
#   PROJECTX_USERNAME=...
#   PROJECTX_API_KEY=...

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
            # Docs show accept: text/plain, but response is JSON. Keep doc-aligned.
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


def contract_available(token: str, live: bool) -> dict:
    payload = {"live": live}
    return post_json(token, "/api/Contract/available", payload, f"CONTRACT_AVAILABLE(live={live})")


def print_contracts(contracts: list, max_n: int = 15) -> None:
    for c in contracts[:max_n]:
        # Fields vary slightly; print defensively
        cid = c.get("id")
        name = c.get("name")
        desc = c.get("description")
        active = c.get("activeContract")
        symbol_id = c.get("symbolId")
        exchange = c.get("exchange")
        print(f"- id={cid} name={name} desc={desc} active={active} symbolId={symbol_id} exch={exchange}")


def normalize_contracts(data: dict) -> list:
    if not isinstance(data, dict):
        return []
    contracts = data.get("contracts")
    if isinstance(contracts, list):
        return contracts
    return []


def looks_like_problem_details(data: dict) -> bool:
    # RFC7807-ish: {title, status, errors, ...}
    return isinstance(data, dict) and ("title" in data and "status" in data and "errors" in data)


def main() -> int:
    username = must_get("PROJECTX_USERNAME")
    api_key = must_get("PROJECTX_API_KEY")
    symbol = (os.getenv("SYMBOL") or "MNQ").strip().upper()

    print(f"BASE_URL = {BASE_URL}")
    print(f"SYMBOL   = {symbol}")

    token = login(username, api_key)
    print("✅ Login OK")

    found_any = False
    best_live = None
    best_contracts = []

    # 1) Try search with live=False then live=True
    for live in (False, True):
        data = contract_search(token, symbol, live=live)

        if looks_like_problem_details(data):
            # Validation-style response; dump and continue
            print(f"\n⚠️ Contract search returned validation/problem response for live={live}:")
            print(json.dumps(data, indent=2)[:2000])
            continue

        contracts = normalize_contracts(data)
        print(f"\n✅ CONTRACT_SEARCH live={live}: {len(contracts)} result(s)")
        if contracts:
            print_contracts(contracts)
            found_any = True
            best_live = live
            best_contracts = contracts
            break
        else:
            # Still useful to print the success flags if present
            if isinstance(data, dict):
                print(f"[DEBUG] success={data.get('success')} errorCode={data.get('errorCode')} errorMessage={data.get('errorMessage')}")

    if found_any:
        print("\n🎯 Next step:")
        print("Use one of the returned contract 'id' values for market data / order routing calls.")
        print(f"(These results came from live={best_live}.)")
        return 0

    # 2) If search returned nothing, inspect available contracts (entitlements/catalog)
    print("\n⚠️ No contracts found via search. Inspecting /api/Contract/available for catalog/entitlements...")

    for live in (False, True):
        data = contract_available(token, live=live)

        if looks_like_problem_details(data):
            print(f"\n⚠️ Contract available returned validation/problem response for live={live}:")
            print(json.dumps(data, indent=2)[:2000])
            continue

        contracts = normalize_contracts(data)
        print(f"\n✅ CONTRACT_AVAILABLE live={live}: {len(contracts)} contract(s)")
        if contracts:
            print("Showing first 20:")
            print_contracts(contracts, max_n=20)
            print("\n🔎 Interpretation:")
            print("- If CONTRACT_AVAILABLE has contracts but CONTRACT_SEARCH had 0 results, your searchText may not match naming.")
            print("  Try SYMBOL=NQ, SYMBOL=MNQU5, SYMBOL=Micro, or SYMBOL=Nasdaq and rerun.")
            return 0

        if isinstance(data, dict):
            print(f"[DEBUG] success={data.get('success')} errorCode={data.get('errorCode')} errorMessage={data.get('errorMessage')}")

    print("\n❌ Still no contracts available (both live=False and live=True).")
    print("This usually indicates an account permission / market-data entitlement issue for this API key/environment, not a payload/code issue.")
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
