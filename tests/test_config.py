import os

from trading_algo.config.env import env_float
from trading_algo.config.settings import load_runtime_config


def test_load_runtime_config(monkeypatch):
    monkeypatch.setenv("BROKER", "projectx")
    monkeypatch.delenv("BROKER_BASE_URL", raising=False)
    monkeypatch.delenv("BROKER_USERNAME", raising=False)
    monkeypatch.delenv("BROKER_API_KEY", raising=False)
    monkeypatch.delenv("BROKER_USER_HUB_URL", raising=False)
    monkeypatch.delenv("BROKER_MARKET_HUB_URL", raising=False)
    monkeypatch.setenv("PROJECTX_BASE_URL", "https://demo.example.com")
    monkeypatch.setenv("PROJECTX_USERNAME", "u")
    monkeypatch.setenv("PROJECTX_API_KEY", "k")
    monkeypatch.setenv("ACCOUNT_ID", "100")
    monkeypatch.setenv("RTC_USER_HUB_URL", "https://demo.example.com/hubs/user")
    monkeypatch.setenv("RTC_MARKET_HUB_URL", "https://demo.example.com/hubs/market")
    monkeypatch.setenv("SYMBOL", "MNQ")
    monkeypatch.setenv("LIVE", "false")
    monkeypatch.setenv("SIDE", "0")
    monkeypatch.setenv("SIZE", "1")
    monkeypatch.setenv("SL_TICKS", "40")
    monkeypatch.setenv("TP_TICKS", "80")
    cfg = load_runtime_config()
    assert cfg.broker == "projectx"
    assert cfg.base_url == "https://demo.example.com"
    assert cfg.account_id == 100
    assert cfg.symbol == "MNQ"


def test_load_runtime_config_broker_alias_vars(monkeypatch):
    monkeypatch.setenv("BROKER", "projectx")
    monkeypatch.setenv("BROKER_BASE_URL", "https://broker.example.com")
    monkeypatch.setenv("BROKER_USERNAME", "broker_u")
    monkeypatch.setenv("BROKER_API_KEY", "broker_k")
    monkeypatch.setenv("ACCOUNT_ID", "101")
    monkeypatch.setenv("BROKER_USER_HUB_URL", "https://broker.example.com/hubs/user")
    monkeypatch.setenv("BROKER_MARKET_HUB_URL", "https://broker.example.com/hubs/market")
    monkeypatch.setenv("SYMBOL", "MES")
    monkeypatch.setenv("LIVE", "true")
    monkeypatch.setenv("SIDE", "1")
    monkeypatch.setenv("SIZE", "2")
    monkeypatch.setenv("SL_TICKS", "20")
    monkeypatch.setenv("TP_TICKS", "30")
    cfg = load_runtime_config()
    assert cfg.base_url == "https://broker.example.com"
    assert cfg.username == "broker_u"
    assert cfg.api_key == "broker_k"
    assert cfg.account_id == 101
    assert cfg.user_hub_url == "https://broker.example.com/hubs/user"
    assert cfg.market_hub_url == "https://broker.example.com/hubs/market"
    assert cfg.symbol == "MES"
    assert cfg.live is True


def test_load_runtime_config_side_text_alias(monkeypatch):
    monkeypatch.setenv("BROKER", "projectx")
    monkeypatch.setenv("BROKER_BASE_URL", "https://broker.example.com")
    monkeypatch.setenv("BROKER_USERNAME", "u")
    monkeypatch.setenv("BROKER_API_KEY", "k")
    monkeypatch.setenv("ACCOUNT_ID", "101")
    monkeypatch.setenv("BROKER_USER_HUB_URL", "https://broker.example.com/hubs/user")
    monkeypatch.setenv("BROKER_MARKET_HUB_URL", "https://broker.example.com/hubs/market")
    monkeypatch.setenv("SYMBOL", "MES")
    monkeypatch.setenv("LIVE", "false")
    monkeypatch.setenv("SIDE", "LONG")
    monkeypatch.setenv("SIZE", "1")
    monkeypatch.setenv("SL_TICKS", "10")
    monkeypatch.setenv("TP_TICKS", "20")
    cfg = load_runtime_config()
    assert cfg.side == 0


def test_env_float_parses_decimal(monkeypatch):
    monkeypatch.setenv("TEST_FLOAT", "1.25")
    assert env_float("TEST_FLOAT", 0.0) == 1.25
    monkeypatch.delenv("TEST_FLOAT")
    assert env_float("TEST_FLOAT", 2.5) == 2.5
