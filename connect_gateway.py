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

BASE_URL = (os.getenv("PROJECTX_BASE_URL") or "https://gateway-api-demo.s2f.projectx.com").rstrip("/")

def login_with_api_key(username: str, api_key: str) -> str:
    # POST /api/Auth/loginKey
    url = f"{BASE_URL}/api/Auth/loginKey"
    r = requests.post(
        url,
        headers={"accept": "text/plain", "Content-Type": "application/json"},
        json={"userName": username, "apiKey": api_key},
        timeout=30,
    )

    try:
        data = r.json()
    except Exception:
        raise RuntimeError(f"Login failed: HTTP {r.status_code}, non-JSON response:\n{r.text}")

    if not data.get("success"):
        raise RuntimeError(f"Login failed: HTTP {r.status_code}, response:\n{data}")

    token = data.get("token")
    if not token:
        raise RuntimeError(f"Login response missing token:\n{data}")

    return token

def search_accounts(token: str, only_active: bool = True) -> dict:
    # POST /api/Account/search
    url = f"{BASE_URL}/api/Account/search"
    headers = {
        "accept": "text/plain",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    r = requests.post(url, headers=headers, json={"onlyActiveAccounts": only_active}, timeout=30)

    try:
        data = r.json()
    except Exception:
        raise RuntimeError(f"Account search failed: HTTP {r.status_code}, non-JSON response:\n{r.text}")

    if r.status_code == 401:
        raise RuntimeError("401 Unauthorized on account search. Token was not accepted.")

    if not data.get("success"):
        raise RuntimeError(f"Account search failed: HTTP {r.status_code}, response:\n{data}")

    return data

def main():
    username = must_get("PROJECTX_USERNAME")
    api_key = must_get("PROJECTX_API_KEY")

    print(f"BASE_URL = {BASE_URL}")
    print("Logging in with API key...")

    token = login_with_api_key(username, api_key)
    print("✅ Login success. Token acquired.")

    print("Fetching accounts...")
    data = search_accounts(token, only_active=True)

    accounts = data.get("accounts", [])
    print(f"✅ Accounts found: {len(accounts)}")
    for a in accounts:
        print(f"- id={a.get('id')} name={a.get('name')} balance={a.get('balance')} canTrade={a.get('canTrade')}")

    print("\n🛑 BOT_ENABLED is still off. No orders placed.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
