import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv()

def must_get(name: str) -> str:
    v = (os.getenv(name) or "").strip().strip('"').strip("'")
    if not v:
        raise RuntimeError(f"Missing {name} in .env")
    return v

BASE_URL = must_get("PROJECTX_BASE_URL").rstrip("/")
ACCOUNT_ID = int(must_get("PROJECTX_ACCOUNT_ID"))

def safe_json(resp: requests.Response, label: str) -> dict:
    ct = (resp.headers.get("content-type") or "").lower()
    text = resp.text or ""
    print(f"[DEBUG] {label} status={resp.status_code} content-type={ct} len={len(text)}")
    if not text.strip():
        raise RuntimeError(f"{label} returned EMPTY body (status {resp.status_code}).")
    try:
        return resp.json()
    except Exception as e:
        raise RuntimeError(f"{label} returned non-JSON: {e}. First200={text[:200]!r}")

def login(username: str, api_key: str) -> str:
    url = f"{BASE_URL}/api/Auth/loginKey"
    r = requests.post(
        url,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        json={"userName": username, "apiKey": api_key},
        timeout=30,
    )
    data = safe_json(r, "LOGIN")
    if not data.get("success") or not data.get("token"):
        raise RuntimeError(f"Login failed: HTTP {r.status_code} {data}")
    return data["token"]

def search_accounts(token: str):
    url = f"{BASE_URL}/api/Account/search"
    r = requests.post(
        url,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        json={"onlyActiveAccounts": True},
        timeout=30,
    )
    data = safe_json(r, "ACCOUNT_SEARCH")
    if not data.get("success"):
        raise RuntimeError(f"Account search failed: HTTP {r.status_code} {data}")
    return data.get("accounts", [])

def main():
    username = must_get("PROJECTX_USERNAME")
    api_key = must_get("PROJECTX_API_KEY")

    print(f"BASE_URL   = {BASE_URL}")
    print(f"ACCOUNT_ID = {ACCOUNT_ID}")

    token = login(username, api_key)
    print("✅ Login OK")

    accounts = search_accounts(token)
    print(f"✅ Found {len(accounts)} active accounts")

    target = next((a for a in accounts if int(a.get("id")) == ACCOUNT_ID), None)
    if not target:
        print("❌ ACCOUNT_ID not found among active accounts:", file=sys.stderr)
        for a in accounts:
            print(f"- id={a.get('id')} name={a.get('name')}", file=sys.stderr)
        sys.exit(1)

    print("\n✅ Selected account:")
    print(f"  id       = {target.get('id')}")
    print(f"  name     = {target.get('name')}")
    print(f"  balance  = {target.get('balance')}")
    print(f"  canTrade = {target.get('canTrade')}")

    if "PRAC" in str(target.get("name","")).upper():
        print("\n✅ This looks like a Practice (PRAC) account. Good.")
    else:
        print("\n⚠️ WARNING: This does NOT look like a Practice (PRAC) account. Stop now.", file=sys.stderr)

    print("\n🛑 BOT_ENABLED is still off. No orders placed.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
