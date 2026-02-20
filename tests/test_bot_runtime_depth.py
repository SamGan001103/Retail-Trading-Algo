from trading_algo.runtime.bot_runtime import _depth_payload_available, _drain_candidate_events


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
