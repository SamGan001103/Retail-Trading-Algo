import os

from trading_algo.config.settings import load_runtime_config


def test_load_runtime_config(monkeypatch):
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
    assert cfg.base_url == "https://demo.example.com"
    assert cfg.account_id == 100
    assert cfg.symbol == "MNQ"
