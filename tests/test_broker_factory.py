from trading_algo.broker import ProjectXBrokerAdapter, broker_from_runtime_config
from trading_algo.config.settings import load_runtime_config


def test_broker_factory_projectx(monkeypatch):
    monkeypatch.setenv("BROKER", "projectx")
    monkeypatch.setenv("BROKER_BASE_URL", "https://demo.example.com")
    monkeypatch.setenv("BROKER_USERNAME", "u")
    monkeypatch.setenv("BROKER_API_KEY", "k")
    monkeypatch.setenv("ACCOUNT_ID", "100")
    monkeypatch.setenv("BROKER_USER_HUB_URL", "https://demo.example.com/hubs/user")
    monkeypatch.setenv("BROKER_MARKET_HUB_URL", "https://demo.example.com/hubs/market")
    monkeypatch.setenv("SYMBOL", "MNQ")
    monkeypatch.setenv("LIVE", "false")
    monkeypatch.setenv("SIDE", "0")
    monkeypatch.setenv("SIZE", "1")
    monkeypatch.setenv("SL_TICKS", "40")
    monkeypatch.setenv("TP_TICKS", "80")
    monkeypatch.setenv("LOOP_SEC", "1")
    monkeypatch.setenv("EXIT_GRACE_SEC", "5")
    cfg = load_runtime_config()
    broker = broker_from_runtime_config(cfg)
    assert isinstance(broker, ProjectXBrokerAdapter)
