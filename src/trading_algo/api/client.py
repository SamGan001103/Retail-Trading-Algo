from __future__ import annotations

import threading
import time
from typing import Any

import requests
from trading_algo.config.env import env_bool


DEBUG_HTTP = env_bool("DEBUG_HTTP", False)


class ProjectXClient:
    def __init__(self, base_url: str, username: str, api_key: str, token_ttl_sec: int = 25 * 60):
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.api_key = api_key
        self.token_ttl_sec = int(token_ttl_sec)
        self._token_cache: str | None = None
        self._token_expiry_ts: float = 0.0
        self._token_lock = threading.Lock()
        self._session = requests.Session()

    def _debug_response(self, response: requests.Response, label: str) -> None:
        if not DEBUG_HTTP:
            return
        ct = (response.headers.get("content-type") or "").lower()
        body = response.text or ""
        print(f"[DEBUG] {label} status={response.status_code} content-type={ct} len={len(body)}")
        if body:
            print(f"[DEBUG] {label} body (first 800): {body[:800]!r}")

    def login_key(self, timeout: int = 30) -> str:
        try:
            response = self._session.post(
                f"{self.base_url}/api/Auth/loginKey",
                headers={"Accept": "application/json", "Content-Type": "application/json"},
                json={"userName": self.username, "apiKey": self.api_key},
                timeout=timeout,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Login request failed: {exc}") from exc
        self._debug_response(response, "LOGIN")
        try:
            data = response.json()
        except Exception as exc:
            raise RuntimeError(
                f"Login returned non-JSON response: HTTP {response.status_code} {response.text[:300]}"
            ) from exc
        if not data.get("success") or not data.get("token"):
            raise RuntimeError(f"Login failed: HTTP {response.status_code} {data}")
        return data["token"]

    def token(self) -> str:
        with self._token_lock:
            now = time.time()
            if self._token_cache and now < self._token_expiry_ts:
                return self._token_cache
            token = self.login_key()
            self._token_cache = token
            self._token_expiry_ts = now + self.token_ttl_sec
            return token

    def set_token(self, token: str, expires_at_ts: float | None = None) -> None:
        with self._token_lock:
            self._token_cache = token
            self._token_expiry_ts = expires_at_ts if expires_at_ts is not None else (time.time() + self.token_ttl_sec)

    def post_json(self, path: str, payload: dict[str, Any], label: str, timeout: int = 30) -> dict[str, Any]:
        def _request(auth_token: str) -> requests.Response:
            return self._session.post(
                f"{self.base_url}{path}",
                headers={
                    "accept": "text/plain",
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {auth_token}",
                },
                json=payload,
                timeout=timeout,
            )

        if DEBUG_HTTP:
            print(f"[DEBUG] Sending payload to {path}: {payload}")

        try:
            response = _request(self.token())
        except requests.RequestException as exc:
            raise RuntimeError(f"{label} request failed: {exc}") from exc
        if response.status_code == 401:
            # Token may have expired early; refresh once and retry.
            with self._token_lock:
                self._token_cache = None
                self._token_expiry_ts = 0.0
            try:
                response = _request(self.token())
            except requests.RequestException as exc:
                raise RuntimeError(f"{label} retry request failed after auth refresh: {exc}") from exc

        self._debug_response(response, label)
        try:
            return response.json()
        except Exception as exc:
            raise RuntimeError(
                f"{label} returned non-JSON response: HTTP {response.status_code} {response.text[:300]}"
            ) from exc

    def close(self) -> None:
        self._session.close()
