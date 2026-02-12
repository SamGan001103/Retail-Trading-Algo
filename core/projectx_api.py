# projectx_api.py

import os
import requests

# ------------------------------------------------------------
# Debug control (off by default)
# Enable with: DEBUG_HTTP=true
# ------------------------------------------------------------
DEBUG_HTTP = (os.getenv("DEBUG_HTTP") or "").strip().lower() in (
    "1", "true", "yes", "y", "on"
)


# ------------------------------------------------------------
# Internal debug printer
# ------------------------------------------------------------
def debug_response(resp: requests.Response, label: str) -> None:
    if not DEBUG_HTTP:
        return

    ct = (resp.headers.get("content-type") or "").lower()
    text = resp.text or ""

    print(f"[DEBUG] {label} status={resp.status_code} content-type={ct} len={len(text)}")

    if text:
        print(f"[DEBUG] {label} body (first 800): {text[:800]!r}")


# ------------------------------------------------------------
# Auth
# ------------------------------------------------------------
def login_key(base_url: str, username: str, api_key: str, timeout: int = 30) -> str:
    """
    Login using API key.
    Returns JWT token string.
    """

    url = f"{base_url.rstrip('/')}/api/Auth/loginKey"

    r = requests.post(
        url,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        json={
            "userName": username,
            "apiKey": api_key,
        },
        timeout=timeout,
    )

    debug_response(r, "LOGIN")

    data = r.json()

    if not data.get("success") or not data.get("token"):
        raise RuntimeError(f"Login failed: HTTP {r.status_code} {data}")

    return data["token"]


# ------------------------------------------------------------
# Generic POST wrapper
# ------------------------------------------------------------
def post_json(
    base_url: str,
    token: str,
    path: str,
    payload: dict,
    label: str,
    timeout: int = 30,
) -> dict:
    """
    Generic POST wrapper with JWT bearer auth.
    """

    url = f"{base_url.rstrip('/')}{path}"

    if DEBUG_HTTP:
        print(f"[DEBUG] Sending payload to {path}: {payload}")

    r = requests.post(
        url,
        headers={
            # ProjectX examples sometimes show accept: text/plain
            # but JSON is returned.
            "accept": "text/plain",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        json=payload,
        timeout=timeout,
    )

    debug_response(r, label)

    try:
        return r.json()
    except Exception:
        raise RuntimeError(
            f"{label} returned non-JSON response: HTTP {r.status_code} {r.text[:300]}"
        )
