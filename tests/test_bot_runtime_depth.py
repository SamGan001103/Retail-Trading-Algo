from datetime import datetime, timezone

from trading_algo.runtime.bot_runtime import (
    _depth_payload_available,
    _drain_candidate_events,
    _extract_event_dt,
    _extract_price,
    _strategy_in_session,
)


def test_depth_payload_available_with_levels():
    depth = {"bids": [{"price": 100.0, "size": 10.0}], "asks": [{"price": 100.25, "size": 9.0}]}
    assert _depth_payload_available(depth) is True


def test_depth_payload_available_with_top_sizes_only():
    depth = {"bestBidSize": 12.0, "bestAskSize": 7.0}
    assert _depth_payload_available(depth) is True


def test_depth_payload_unavailable_when_missing_sizes():
    depth = {"bids": [], "asks": []}
    assert _depth_payload_available(depth) is False


def test_drain_candidate_events_returns_latest_entered_candidate_id():
    class _FakeStrategy:
        def drain_candidate_events(self):
            return [
                {"candidate_id": "cand-1", "status": "detected"},
                {"candidate_id": "cand-2", "status": "entered"},
            ]

    class _FakeTelemetry:
        def __init__(self) -> None:
            self.rows = []

        def emit_candidate(self, payload):
            self.rows.append(payload)

    telemetry = _FakeTelemetry()
    entered = _drain_candidate_events(
        _FakeStrategy(),
        telemetry,
        source="tick",
        event_ts="2026-01-01T00:00:00Z",
        context_index=7,
    )
    assert entered == "cand-2"
    assert len(telemetry.rows) == 2
    assert telemetry.rows[1]["candidate_id"] == "cand-2"


def test_extract_event_dt_prefers_newer_quote_timestamp_over_stale_trade():
    quote = {"timestamp": "2026-02-20T14:35:00Z", "bid": 100.0, "ask": 101.0}
    trade = {"timestamp": "2026-02-19T22:00:00Z", "price": 99.0}

    event_dt = _extract_event_dt(quote, trade)
    assert event_dt == datetime(2026, 2, 20, 14, 35, tzinfo=timezone.utc)


def test_extract_price_prefers_quote_mid_when_quote_is_newer_than_trade():
    quote = {"timestamp": "2026-02-20T14:35:00Z", "bid": 100.0, "ask": 101.0}
    trade = {"timestamp": "2026-02-19T22:00:00Z", "price": 99.0}

    price = _extract_price(quote, trade, last_price=None)
    assert price == 100.5


def test_extract_price_prefers_trade_when_trade_is_newer_than_quote():
    quote = {"timestamp": "2026-02-20T14:35:00Z", "bid": 100.0, "ask": 101.0}
    trade = {"timestamp": "2026-02-20T14:35:01Z", "price": 99.0}

    price = _extract_price(quote, trade, last_price=None)
    assert price == 99.0


def test_extract_price_prefers_quote_when_trade_has_no_timestamp():
    quote = {"timestamp": "2026-02-20T14:35:00Z", "bid": 100.0, "ask": 101.0}
    trade = {"price": 99.0}

    price = _extract_price(quote, trade, last_price=None)
    assert price == 100.5


def test_strategy_in_session_uses_strategy_hook():
    class _Strategy:
        def is_in_session(self, ts: str) -> bool:
            return ts.startswith("2026-02-20")

    in_session = _strategy_in_session(_Strategy(), datetime(2026, 2, 20, 14, 0, tzinfo=timezone.utc))
    out_session = _strategy_in_session(_Strategy(), datetime(2026, 2, 21, 14, 0, tzinfo=timezone.utc))
    assert in_session is True
    assert out_session is False


def test_strategy_in_session_returns_none_when_unavailable():
    class _Strategy:
        pass

    assert _strategy_in_session(_Strategy(), datetime(2026, 2, 20, 14, 0, tzinfo=timezone.utc)) is None
