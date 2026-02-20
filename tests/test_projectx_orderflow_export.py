from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from trading_algo.data_export.projectx_orderflow import _read_existing_max_seq, build_orderflow_row


def test_build_orderflow_row_with_depth_and_trade():
    row, reason = build_orderflow_row(
        seq=12,
        quote={"bid": 100.0, "ask": 100.25},
        trade={"price": 100.1, "size": 3, "timestamp": "2026-01-01T14:30:00Z"},
        depth={
            "bestBid": 100.0,
            "bestAsk": 100.25,
            "bestBidSize": 40,
            "bestAskSize": 35,
            "bids": [{"price": 100.0, "size": 40}],
            "asks": [{"price": 100.25, "size": 35}],
        },
        now_utc=datetime(2026, 1, 1, 14, 30, 1, tzinfo=timezone.utc),
        include_depth_json=True,
        require_depth=True,
    )
    assert reason is None
    assert row is not None
    assert row["seq"] == 12
    assert row["timestamp"] == "2026-01-01T14:30:00Z"
    assert row["price"] == 100.1
    assert row["trade_price"] == 100.1
    assert row["bestBidSize"] == 40.0
    assert row["bestAskSize"] == 35.0
    assert str(row["depth_bids"]).startswith("[")
    assert str(row["depth_asks"]).startswith("[")


def test_build_orderflow_row_rejects_when_depth_required_but_missing():
    row, reason = build_orderflow_row(
        seq=1,
        quote={"bid": 100.0, "ask": 100.25},
        trade={"price": 100.1, "size": 1},
        depth=None,
        now_utc=datetime(2026, 1, 1, 14, 30, 1, tzinfo=timezone.utc),
        include_depth_json=True,
        require_depth=True,
    )
    assert row is None
    assert reason == "no-depth"


def test_build_orderflow_row_uses_epoch_millis_timestamp():
    row, reason = build_orderflow_row(
        seq=1,
        quote={"bid": 100.0, "ask": 100.25},
        trade={"price": 100.1, "size": 1, "timestamp": 1767277800000},
        depth={"bestBidSize": 10, "bestAskSize": 11},
        now_utc=datetime(2026, 1, 1, 14, 30, 1, tzinfo=timezone.utc),
        include_depth_json=False,
        require_depth=True,
    )
    assert reason is None
    assert row is not None
    assert row["timestamp"].endswith("Z")


def test_read_existing_max_seq(tmp_path: Path):
    csv_path = tmp_path / "capture.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "seq", "price"])
        writer.writeheader()
        writer.writerow({"timestamp": "2026-01-01T00:00:00Z", "seq": "10", "price": "100"})
        writer.writerow({"timestamp": "2026-01-01T00:00:01Z", "seq": "11", "price": "100.1"})

    assert _read_existing_max_seq(csv_path) == 11
