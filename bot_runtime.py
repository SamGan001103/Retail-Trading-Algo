# bot_runtime.py  (add these changes to prove why it says FLAT + make it accurate)
# - Prints what realtime is seeing (sample keys + contractId)
# - After placing an order, verifies via REST that orders/positions exist
# - Logs a warning if realtime is not updating

from __future__ import annotations

import os
import time
from dotenv import load_dotenv

from execution_engine import ExecutionEngine, BUY
from projectx_api import login_key, post_json
from realtime_client import RealtimeClient

load_dotenv()


def must_env(name: str) -> str:
    v = (os.getenv(name) or "").strip().strip('"').strip("'")
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v


def env_int(name: str, default: int) -> int:
    v = (os.getenv(name) or "").strip()
    return int(v) if v else default


def env_bool(name: str, default: bool = False) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "y", "on")


def resolve_contract_id(base_url: str, username: str, api_key: str, symbol: str, live: bool) -> str:
    token = login_key(base_url, username, api_key)
    data = post_json(
        base_url,
        token,
        "/api/Contract/search",
        {"live": bool(live), "searchText": symbol},
        "CONTRACT_SEARCH",
    )
    if not data.get("success") or not data.get("contracts"):
        raise RuntimeError(f"Contract search failed: {data}")
    return data["contracts"][0]["id"]


def _filter_by_contract(items: list[dict], contract_id: str) -> list[dict]:
    # NOTE: if realtime payload uses int contractId but you compare to string (e.g. "CON.F.US..."),
    # this will always fail. We print samples below to confirm.
    return [x for x in items if str(x.get("contractId")) == str(contract_id)]


def _rest_verify(base_url: str, username: str, api_key: str, account_id: int) -> None:
    """
    Sanity check what the broker thinks exists.
    If endpoints differ for your gateway, paste the error and we’ll align them.
    """
    token = login_key(base_url, username, api_key)

    # These endpoint names may vary across ProjectX versions; adjust if your API uses different routes.
    orders = post_json(base_url, token, "/api/Order/search", {"accountId": account_id}, "ORDER_SEARCH")
    pos = post_json(base_url, token, "/api/Position/search", {"accountId": account_id}, "POSITION_SEARCH")

    orders_list = orders.get("orders") or []
    pos_list = pos.get("positions") or []

    print(f"🧾 REST VERIFY: orders={len(orders_list)} positions={len(pos_list)}")
    if orders_list:
        o0 = orders_list[0]
        print("🧾 REST ORDER[0] keys:", list(o0.keys()))
        print("🧾 REST ORDER[0] contractId:", o0.get("contractId"), "status:", o0.get("status"))
    if pos_list:
        p0 = pos_list[0]
        print("🧾 REST POS[0] keys:", list(p0.keys()))
        print("🧾 REST POS[0] contractId:", p0.get("contractId"), "qty:", p0.get("quantity") or p0.get("netQuantity"))


def main() -> None:
    base_url = must_env("PROJECTX_BASE_URL").rstrip("/")
    username = must_env("PROJECTX_USERNAME")
    api_key = must_env("PROJECTX_API_KEY")
    account_id = int(must_env("ACCOUNT_ID"))

    rtc_user_hub = must_env("RTC_USER_HUB_URL")
    rtc_market_hub = must_env("RTC_MARKET_HUB_URL")

    symbol = (os.getenv("SYMBOL") or "MNQ").strip().upper()
    live = env_bool("LIVE", False)

    side = env_int("SIDE", BUY)
    size = env_int("SIZE", 1)
    sl_ticks = env_int("SL_TICKS", 40)
    tp_ticks = env_int("TP_TICKS", 80)

    loop_sec = env_int("LOOP_SEC", 1)
    exit_grace_sec = env_int("EXIT_GRACE_SEC", 5)

    flatten_on_start = env_bool("FLATTEN_ON_START", False)
    trade_on_start = env_bool("TRADE_ON_START", False)

    eng = ExecutionEngine(base_url, username, api_key, account_id)

    contract_id = resolve_contract_id(base_url, username, api_key, symbol, live)
    print(f"✅ contract_id={contract_id} for symbol={symbol} live={live}")

    rt = RealtimeClient(
        base_url=base_url,
        username=username,
        api_key=api_key,
        account_id=account_id,
        user_hub_url=rtc_user_hub,
        market_hub_url=rtc_market_hub,
    )
    rt.start(contract_id=contract_id)
    print("📡 Realtime streaming started (user + market).")

    if flatten_on_start:
        c, p = eng.flatten()
        print(f"🧹 Flatten on start: cancel_attempted={c} close_attempted={p}")

    traded_once = False
    was_in_pos = False
    exit_grace_until = 0.0

    # Debug latch: print realtime samples once when they first appear
    printed_rt_samples = False

    print("🚦 Bot runtime started. Press Ctrl+C to stop.")

    try:
        while True:
            orders_all, positions_all = rt.snapshot()

            # ---- DEBUG: show what realtime is actually returning ----
            if (not printed_rt_samples) and (orders_all or positions_all):
                printed_rt_samples = True
                if orders_all:
                    print("🔎 RT ORDER[0] keys:", list(orders_all[0].keys()))
                    print("🔎 RT ORDER[0] contractId:", orders_all[0].get("contractId"))
                if positions_all:
                    print("🔎 RT POS[0] keys:", list(positions_all[0].keys()))
                    print("🔎 RT POS[0] contractId:", positions_all[0].get("contractId"))

            # Filter to our contract only
            orders = _filter_by_contract(orders_all, contract_id)
            positions = _filter_by_contract(positions_all, contract_id)

            in_pos = len(positions) > 0
            now = time.time()

            if was_in_pos and not in_pos:
                exit_grace_until = now + float(exit_grace_sec)
            was_in_pos = in_pos
            in_exit_grace = now < exit_grace_until

            # --------- STATE MACHINE ---------
            if not in_pos:
                if orders:
                    if in_exit_grace:
                        print(f"[HEARTBEAT] FLAT (exit grace) waiting for OCO cancel. open_orders={len(orders)}")
                    else:
                        print(f"[HEARTBEAT] FLAT but stale orders exist: {len(orders)} (blocking entry)")
                else:
                    print("[HEARTBEAT] FLAT and clean (can enter)")

                # One-trade latch
                if trade_on_start and (not traded_once) and (not orders) and (not in_exit_grace):
                    resp = eng.place_market_with_brackets(
                        contract_id=contract_id,
                        side=side,
                        size=size,
                        sl_ticks_abs=sl_ticks,
                        tp_ticks_abs=tp_ticks,
                    )
                    print("📤 place_market_with_brackets resp:", resp)

                    # ---- REST sanity check right after sending ----
                    try:
                        _rest_verify(base_url, username, api_key, account_id)
                    except Exception as e:
                        print("⚠️ REST verify failed:", repr(e))

                    if resp.get("success"):
                        traded_once = True
                        print("✅ Trade placed (trade-once latch set).")
                        print("⚠️ If HEARTBEAT still says FLAT, realtime orders/positions are not updating or contractId mismatch.")
                    else:
                        print("❌ Trade rejected:", resp)

            else:
                print(f"[HEARTBEAT] IN POSITION. Working exit orders: {len(orders)}")

                if len(orders) < 2:
                    print("🚨 Missing bracket(s) while in position → flattening.")
                    c, p = eng.flatten()
                    print(f"🧹 Flatten: cancel_attempted={c} close_attempted={p}")

            time.sleep(loop_sec)

    except KeyboardInterrupt:
        print("\n🛑 Stopping bot (KeyboardInterrupt).")
    finally:
        try:
            rt.stop()
            print("📡 Realtime streaming stopped.")
        except Exception as e:
            print(f"⚠️ Error stopping realtime: {e}")


if __name__ == "__main__":
    main()
