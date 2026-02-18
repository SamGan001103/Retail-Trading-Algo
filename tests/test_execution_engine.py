from trading_algo.execution import BUY, SELL, ExecutionEngine, sign_brackets


class FakeClient:
    def __init__(self):
        self.responses = {}
        self.calls = []

    def post_json(self, path, payload, label, timeout=30):  # noqa: ARG002
        self.calls.append((path, payload, label))
        return self.responses.get(path, {"success": True})


def test_sign_brackets_long_short():
    long_b = sign_brackets(BUY, 40, 80)
    short_b = sign_brackets(SELL, 40, 80)
    assert long_b.sl_ticks == -40 and long_b.tp_ticks == 80
    assert short_b.sl_ticks == 40 and short_b.tp_ticks == -80


def test_place_market_with_brackets_payload():
    client = FakeClient()
    engine = ExecutionEngine(client, account_id=123)
    engine.place_market_with_brackets("CON123", BUY, 2, 10, 20)
    path, payload, label = client.calls[-1]
    assert path == "/api/Order/place"
    assert label == "ORDER_PLACE"
    assert payload["accountId"] == 123
    assert payload["contractId"] == "CON123"
    assert payload["stopLossBracket"]["ticks"] == -10
    assert payload["takeProfitBracket"]["ticks"] == 20


def test_can_enter_trade_when_flat_and_clean():
    client = FakeClient()
    client.responses["/api/Position/searchOpen"] = {"success": True, "positions": []}
    client.responses["/api/Order/searchOpen"] = {"success": True, "orders": []}
    engine = ExecutionEngine(client, account_id=123)
    ok, reason = engine.can_enter_trade()
    assert ok is True
    assert reason == "OK"
