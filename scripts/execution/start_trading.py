import os

from trading_algo.config import env_bool, load_runtime_config
from trading_algo.runtime.bot_runtime import run


def main() -> None:
    cfg = load_runtime_config()
    enabled = env_bool("BOT_ENABLED", False)
    environment = (os.getenv("TRADING_ENVIRONMENT") or "DEMO").strip()

    print(f"TRADING_ENVIRONMENT = {environment}")
    print(f"BOT_ENABLED        = {enabled}")
    print(f"SYMBOL             = {cfg.symbol}")
    print(f"ACCOUNT_ID         = {cfg.account_id}")
    print(f"LIVE               = {cfg.live}")
    print(f"TRADE_ON_START     = {cfg.trade_on_start}")

    if not enabled:
        print("BOT_ENABLED=0 -> Trading disabled.")
        return

    # Master launcher for runtime trading loop (WIP, strategy not finalized).
    run(cfg)


if __name__ == "__main__":
    main()
