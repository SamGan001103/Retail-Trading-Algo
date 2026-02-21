from trading_algo.broker.projectx_realtime import JsonHubProtocolRS


def test_json_hub_protocol_rs_parses_valid_ping_frame():
    protocol = JsonHubProtocolRS()

    messages = protocol.parse_messages("{\"type\":6}\x1e")

    assert len(messages) == 1


def test_json_hub_protocol_rs_ignores_malformed_frame_and_continues():
    parse_errors: list[tuple[str, int]] = []

    def _on_error(exc: Exception, payload_size: int) -> None:
        parse_errors.append((exc.__class__.__name__, payload_size))

    protocol = JsonHubProtocolRS(on_parse_error=_on_error)
    messages = protocol.parse_messages("{bad}\x1e{\"type\":6}\x1e")

    assert len(messages) == 1
    assert parse_errors
    assert parse_errors[0][1] == len("{bad}")


def test_json_hub_protocol_rs_buffers_partial_frame():
    protocol = JsonHubProtocolRS()
    first = protocol.parse_messages("{\"type\":6")
    second = protocol.parse_messages("}\x1e")

    assert first == []
    assert len(second) == 1
