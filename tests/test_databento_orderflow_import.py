from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

from trading_algo.data_export.databento_orderflow import _record_to_orderflow_row


@dataclass(frozen=True)
class _FakeAction:
    name: str


@dataclass(frozen=True)
class _FakeLevel:
    bid_px: int
    bid_sz: int
    ask_px: int
    ask_sz: int


@dataclass(frozen=True)
class _FakeRecord:
    ts_event: int
    sequence: int | None
    action: _FakeAction
    price: int
    size: int
    levels: list[_FakeLevel]


def _ns(dt: datetime) -> int:
    return int(dt.timestamp()) * 1_000_000_000


def test_record_to_orderflow_row_trade_action_uses_trade_fields():
    record = _FakeRecord(
        ts_event=_ns(datetime(2026, 2, 18, 14, 30, tzinfo=timezone.utc)),
        sequence=7,
        action=_FakeAction(name="TRADE"),
        price=21_850_250_000_000,
        size=3,
        levels=[
            _FakeLevel(
                bid_px=21_850_000_000_000,
                bid_sz=40,
                ask_px=21_850_500_000_000,
                ask_sz=35,
            )
        ],
    )
    row = _record_to_orderflow_row(record, fallback_seq=1, include_depth_json=True)
    assert row is not None
    assert row["timestamp"] == "2026-02-18T14:30:00Z"
    assert row["seq"] == 7
    assert row["price"] == 21850.25
    assert row["trade_price"] == 21850.25
    assert row["trade_size"] == 3.0
    depth_bids = json.loads(str(row["depth_bids"]))
    depth_asks = json.loads(str(row["depth_asks"]))
    assert depth_bids[0]["price"] == 21850.0
    assert depth_asks[0]["price"] == 21850.5


def test_record_to_orderflow_row_non_trade_action_zeros_trade_size():
    record = _FakeRecord(
        ts_event=_ns(datetime(2026, 2, 18, 14, 30, 1, tzinfo=timezone.utc)),
        sequence=8,
        action=_FakeAction(name="ADD"),
        price=99_999_000_000_000,
        size=11,
        levels=[
            _FakeLevel(
                bid_px=21_900_000_000_000,
                bid_sz=22,
                ask_px=21_900_500_000_000,
                ask_sz=19,
            )
        ],
    )
    row = _record_to_orderflow_row(record, fallback_seq=999, include_depth_json=False)
    assert row is not None
    assert row["seq"] == 8
    assert row["price"] == 21900.25
    assert row["trade_price"] == ""
    assert row["trade_size"] == 0.0
    assert row["depth_bids"] == ""
    assert row["depth_asks"] == ""


def test_record_to_orderflow_row_fallback_seq_when_sequence_missing():
    record = _FakeRecord(
        ts_event=_ns(datetime(2026, 2, 18, 14, 30, 2, tzinfo=timezone.utc)),
        sequence=None,
        action=_FakeAction(name="TRADE"),
        price=21_800_000_000_000,
        size=1,
        levels=[
            _FakeLevel(
                bid_px=21_799_750_000_000,
                bid_sz=10,
                ask_px=21_800_250_000_000,
                ask_sz=9,
            )
        ],
    )
    row = _record_to_orderflow_row(record, fallback_seq=12345, include_depth_json=True)
    assert row is not None
    assert row["seq"] == 12345
