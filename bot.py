import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

def must_get(name: str) -> str:
    v = (os.getenv(name) or "").strip().strip('"').strip("'")
    if not v:
        raise RuntimeError(f"Missing required env var: {name} (set it in .env)")
    return v

@dataclass(frozen=True)
class Config:
    username: str
    api_key: str
    trading_environment: str
    bot_enabled: bool

def load_config() -> Config:
    return Config(
        username=must_get("PROJECTX_USERNAME"),
        api_key=must_get("PROJECTX_API_KEY"),
        trading_environment=(os.getenv("TRADING_ENVIRONMENT") or "DEMO").strip(),
        bot_enabled=((os.getenv("BOT_ENABLED") or "0").strip() == "1"),
    )

def main():
    cfg = load_config()
    print("✅ .env loaded successfully")
    print(f"TRADING_ENVIRONMENT = {cfg.trading_environment}")
    print(f"BOT_ENABLED        = {cfg.bot_enabled}")

    if not cfg.bot_enabled:
        print("🛑 BOT_ENABLED=0 → Trading disabled (correct for now).")
        return

    print("⚠️ BOT_ENABLED=1 but trading logic not implemented yet.")

if __name__ == "__main__":
    main()
