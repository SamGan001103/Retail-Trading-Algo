import os

from trading_algo.config import env_bool
from trading_algo.runtime.bot_runtime import main as run_bot


def main() -> None:
    enabled = env_bool("BOT_ENABLED", False)
    environment = (os.getenv("TRADING_ENVIRONMENT") or "DEMO").strip()
    print(f"TRADING_ENVIRONMENT = {environment}")
    print(f"BOT_ENABLED        = {enabled}")
    if not enabled:
        print("BOT_ENABLED=0 -> Trading disabled.")
        return
    run_bot()


if __name__ == "__main__":
    main()
