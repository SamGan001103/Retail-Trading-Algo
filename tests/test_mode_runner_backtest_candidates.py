from datetime import datetime, timezone
from pathlib import Path

import pytest

from trading_algo.backtest import BacktestConfig, OrderFlowParquetScan, OrderFlowTick
from trading_algo.runtime.mode_runner import (
    _BacktestMatrixTracker,
    _BacktestMatrixParquetWriter,
    _BacktestSummaryParquetWriter,
    _BacktestCandidateParquetWriter,
    _assess_backtest_health,
    _apply_shadow_ml,
    _build_backtest_scenarios,
    _build_backtest_shadow_gate,
    _build_walk_forward_windows,
    _ensure_orderflow_backtest_dataset,
    _load_major_news_blackouts,
    _latest_months_window,
    _latest_months_window_ticks,
    _validate_orderflow_backtest_preflight_parquet,
)
from trading_algo.strategy import MarketBar


def _bar(ts: str) -> MarketBar:
    return MarketBar(ts=ts, open=100.0, high=101.0, low=99.0, close=100.0, volume=1.0)


def test_latest_months_window_keeps_most_recent_six_months():
    bars = [
        _bar("2025-01-01T00:00:00Z"),
        _bar("2025-07-01T00:00:00Z"),
        _bar("2026-01-01T00:00:00Z"),
    ]

    selected, start_dt, end_dt = _latest_months_window(bars, 6)

    assert start_dt is not None
    assert end_dt is not None
    assert [bar.ts for bar in selected] == [
        "2025-07-01T00:00:00Z",
        "2026-01-01T00:00:00Z",
    ]


def _read_parquet_rows(dataset_dir: Path) -> list[dict]:
    ds = pytest.importorskip("pyarrow.dataset")

    table = ds.dataset(str(dataset_dir), format="parquet").to_table()
    columns = table.to_pydict()
    names = list(columns.keys())
    rows = len(columns[names[0]]) if names else 0
    out: list[dict] = []
    for i in range(rows):
        out.append({name: columns[name][i] for name in names})
    return out


def test_backtest_candidate_parquet_writer_appends_rows(tmp_path: Path):
    pytest.importorskip("pyarrow")
    parquet_path = tmp_path / "backtest_candidates.parquet"
    writer = _BacktestCandidateParquetWriter(
        str(parquet_path),
        run_id="run-1",
        strategy="ny_structure",
        scenario_id="base",
        window_id="latest_6m",
        window_months=6,
        window_start_utc=datetime(2025, 7, 1, tzinfo=timezone.utc),
        window_end_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    writer.append(
        {
            "event_name": "strategy_candidate",
            "bar_index": 10,
            "bar_ts": "2026-01-01T00:10:00Z",
            "candidate_id": "cand-1",
            "status": "detected",
            "reason": "setup-ready",
            "side": "buy",
            "setup_index": 10,
            "setup_ts": "2026-01-01T00:10:00Z",
            "confluence_score": 2,
        }
    )
    writer.append(
        {
            "event_name": "strategy_candidate",
            "bar_index": 11,
            "bar_ts": "2026-01-01T00:11:00Z",
            "candidate_id": "cand-2",
            "status": "entered",
            "reason": "entry-ny-structure-bar",
            "side": "sell",
            "setup_index": 11,
            "setup_ts": "2026-01-01T00:11:00Z",
            "confluence_score": 3,
        }
    )
    writer.close()

    rows = _read_parquet_rows(tmp_path / "backtest_candidates")
    assert len(rows) == 2
    candidate_ids = {str(row.get("candidate_id")) for row in rows}
    assert candidate_ids == {"cand-1", "cand-2"}


def test_backtest_matrix_parquet_writer_appends_rows(tmp_path: Path):
    pytest.importorskip("pyarrow")
    parquet_path = tmp_path / "backtest_candidate_matrix.parquet"
    writer = _BacktestMatrixParquetWriter(str(parquet_path))
    writer.append_rows(
        [
            {
                "run_id": "run-2",
                "strategy": "ny_structure",
                "scenario_id": "base",
                "window_id": "latest_6m",
                "window_months": 6,
                "candidate_id": "cand-new",
            }
        ]
    )
    writer.close()

    rows = _read_parquet_rows(tmp_path / "backtest_candidate_matrix")
    assert len(rows) == 1
    assert rows[0]["candidate_id"] == "cand-new"


def test_backtest_summary_parquet_writer_appends_rows(tmp_path: Path):
    pytest.importorskip("pyarrow")
    parquet_path = tmp_path / "backtest_summary.parquet"
    writer = _BacktestSummaryParquetWriter(str(parquet_path))
    writer.append(
        {
            "run_id": "run-1",
            "strategy": "ny_structure",
            "source_path": "data/parquet/day1",
            "scenario_id": "base",
            "window_id": "latest_6m",
            "window_start_utc": "2026-01-01T00:00:00Z",
            "window_end_utc": "2026-01-02T00:00:00Z",
            "window_months": 1,
            "bars": 1000,
            "num_trades": 22,
            "final_equity": 10012.5,
            "net_pnl": 12.5,
            "return_pct": 0.125,
            "win_rate_pct": 54.0,
            "max_drawdown_pct": 1.2,
            "orderflow_replay": True,
            "news_blackouts": 2,
        }
    )
    writer.close()

    rows = _read_parquet_rows(tmp_path / "backtest_summary")
    assert len(rows) == 1
    assert rows[0]["source_path"] == "data/parquet/day1"
    assert rows[0]["num_trades"] == 22


def test_latest_months_window_ticks_keeps_recent_window():
    ticks = [
        OrderFlowTick(
            ts="2025-01-01T00:00:00Z",
            price=100.0,
            volume=1.0,
            quote=None,
            trade=None,
            depth={"bestBidSize": 1.0, "bestAskSize": 1.0},
        ),
        OrderFlowTick(
            ts="2025-09-01T00:00:00Z",
            price=100.0,
            volume=1.0,
            quote=None,
            trade=None,
            depth={"bestBidSize": 1.0, "bestAskSize": 1.0},
        ),
        OrderFlowTick(
            ts="2026-01-01T00:00:00Z",
            price=100.0,
            volume=1.0,
            quote=None,
            trade=None,
            depth={"bestBidSize": 1.0, "bestAskSize": 1.0},
        ),
    ]
    selected, _, _ = _latest_months_window_ticks(ticks, 6)
    assert [x.ts for x in selected] == [
        "2025-09-01T00:00:00Z",
        "2026-01-01T00:00:00Z",
    ]


def test_ensure_orderflow_backtest_dataset_requires_depth():
    ticks = [
        OrderFlowTick(
            ts="2026-01-01T00:00:00Z",
            price=100.0,
            volume=1.0,
            quote=None,
            trade=None,
            depth={},
        )
    ]
    try:
        _ensure_orderflow_backtest_dataset(ticks, "x.parquet")
        assert False, "expected RuntimeError for missing depth fields"
    except RuntimeError as exc:
        assert "No usable depth data" in str(exc)


def test_load_major_news_blackouts_filters_major_and_merges(tmp_path: Path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    news_path = tmp_path / "news.parquet"
    table = pa.table(
        {
            "timestamp": [
                "2026-01-01T14:30:00Z",
                "2026-01-01T14:40:00Z",
                "2026-01-01T14:45:00Z",
            ],
            "impact": ["high", "low", "3"],
            "currency": ["USD", "USD", "USD"],
            "event": ["ISM", "minor", "NFP"],
        }
    )
    pq.write_table(table, str(news_path))
    intervals = _load_major_news_blackouts(
        str(news_path),
        pre_minutes=10,
        post_minutes=10,
        window_start=datetime(2026, 1, 1, 14, 0, tzinfo=timezone.utc),
        window_end=datetime(2026, 1, 1, 16, 0, tzinfo=timezone.utc),
        major_only=True,
        currencies=("USD",),
    )
    assert len(intervals) == 1
    assert intervals[0][0].isoformat().startswith("2026-01-01T14:20:00")
    assert intervals[0][1].isoformat().startswith("2026-01-01T14:55:00")


def test_backtest_matrix_tracker_captures_trade_outcome():
    tracker = _BacktestMatrixTracker(
        run_id="run-1",
        strategy="ny_structure",
        scenario_id="base",
        window_id="latest_6m",
        window_months=6,
        window_start_utc=datetime(2025, 7, 1, tzinfo=timezone.utc),
        window_end_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
        tick_value=0.5,
    )
    tracker.on_candidate_event(
        {
            "candidate_id": "cand-1",
            "status": "entered",
            "reason": "entry-orderflow-sniper",
            "side": "buy",
            "setup_index": 10,
            "setup_ts": "2026-01-01T14:31:00Z",
            "setup_close": 100.0,
            "confluence_score": 2,
        }
    )
    tracker.on_execution_event(
        {
            "event_name": "tick_enter",
            "candidate_id": "cand-1",
            "bar_index": 10,
            "bar_ts": "2026-01-01T14:31:01Z",
            "reason": "entry-orderflow-sniper",
            "entry_price": 100.25,
            "size": 2,
            "sl_ticks_abs": 10,
            "tp_ticks_abs": 30,
        }
    )
    tracker.on_execution_event(
        {
            "event_name": "tick_exit",
            "candidate_id": "cand-1",
            "bar_index": 11,
            "bar_ts": "2026-01-01T14:32:00Z",
            "reason": "take-profit",
            "exit_price": 101.5,
            "pnl": 7.5,
        }
    )
    rows = tracker.finalize_rows()
    assert len(rows) == 1
    row = rows[0]
    assert row["candidate_id"] == "cand-1"
    assert row["risk_dollars"] == 10.0
    assert row["pnl"] == 7.5
    assert row["win"] == 1
    assert row["loss"] == 0
    assert row["result_label"] == "win"
    assert row["realized_rr"] == 0.75


def test_build_walk_forward_windows_rolls_monthly():
    times = [
        datetime(2025, 1, 1, tzinfo=timezone.utc),
        datetime(2025, 2, 1, tzinfo=timezone.utc),
        datetime(2025, 3, 1, tzinfo=timezone.utc),
        datetime(2025, 4, 1, tzinfo=timezone.utc),
        datetime(2025, 5, 1, tzinfo=timezone.utc),
        datetime(2025, 6, 1, tzinfo=timezone.utc),
        datetime(2025, 7, 1, tzinfo=timezone.utc),
    ]
    windows = _build_walk_forward_windows(
        parsed_times_utc=times,
        window_months=3,
        step_months=1,
        start_utc=None,
        end_utc=None,
    )
    assert [w.window_id for w in windows] == ["wf_001", "wf_002", "wf_003", "wf_004"]


def test_shadow_ml_log_only_when_model_missing(monkeypatch):
    monkeypatch.setenv("BACKTEST_SHADOW_ML_ENABLED", "true")
    monkeypatch.setenv("BACKTEST_SHADOW_ML_MODEL_PATH", "")
    gate = _build_backtest_shadow_gate()
    assert gate is not None
    enriched = _apply_shadow_ml(
        {
            "candidate_id": "cand-1",
            "side": "buy",
            "has_recent_sweep": True,
            "bias_ok": True,
            "continuation": True,
            "reversal": False,
            "equal_levels": True,
            "fib_retracement": True,
            "key_area_proximity": False,
            "confluence_score": 3,
        },
        gate,
    )
    assert enriched["ml_shadow_approved"] is True
    assert str(enriched["ml_shadow_reason"]).startswith("ml-unavailable:")
    assert enriched["ml_shadow_score"] is None


def test_build_backtest_scenarios_includes_sweeps(monkeypatch):
    monkeypatch.setenv("BACKTEST_SENSITIVITY_SWEEP", "true")
    monkeypatch.setenv("BACKTEST_SWEEP_ENTRY_DELAYS", "0,1")
    monkeypatch.setenv("BACKTEST_SWEEP_SLIP_ENTRY_TICKS", "0.0,1.0")
    monkeypatch.setenv("BACKTEST_SWEEP_SPREAD_SLIP_K", "0,2")
    scenarios = _build_backtest_scenarios(
        BacktestConfig(
            initial_cash=10_000.0,
            fee_per_order=1.0,
            slippage_bps=1.0,
            tick_size=0.25,
            max_drawdown_abs=None,
            slip_entry_ticks=0.0,
            slip_stop_ticks=0.0,
            slip_tp_ticks=0.0,
            spread_slip_k=1.0,
            entry_delay_events=1,
        )
    )
    ids = [s.scenario_id for s in scenarios]
    assert "base" in ids
    assert "latency_d1" in ids
    assert "slip_e1" in ids
    assert "spread_k2" in ids


def test_validate_orderflow_preflight_report(monkeypatch):
    scan = OrderFlowParquetScan(
        path="data/parquet/ticks",
        source_files=["data/parquet/ticks/part-00001.parquet"],
        columns=["ts_event", "sequence", "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"],
        rows=2,
        parseable_timestamps=2,
        explicit_tz_timestamps=2,
        quote_rows=2,
        depth_rows=2,
        first_ts_utc=datetime(2026, 1, 1, 14, 30, tzinfo=timezone.utc),
        last_ts_utc=datetime(2026, 1, 1, 14, 30, 1, tzinfo=timezone.utc),
    )
    monkeypatch.setenv("BACKTEST_PREFLIGHT_STRICT", "true")
    monkeypatch.setenv("BACKTEST_PREFLIGHT_MIN_ROWS", "1")
    monkeypatch.setenv("BACKTEST_PREFLIGHT_MIN_SESSION_ROWS", "1")
    monkeypatch.setenv("BACKTEST_PREFLIGHT_MIN_QUOTE_COVERAGE", "0.5")
    monkeypatch.setenv("BACKTEST_PREFLIGHT_MIN_DEPTH_COVERAGE", "0.5")
    monkeypatch.setenv("STRAT_AVOID_NEWS", "false")
    report = _validate_orderflow_backtest_preflight_parquet(
        data_path=scan.path,
        scan=scan,
        window_start=datetime(2026, 1, 1, 14, 0, tzinfo=timezone.utc),
        window_end=datetime(2026, 1, 1, 15, 0, tzinfo=timezone.utc),
        news_blackouts=[],
    )
    assert report["rows"] == 2
    assert report["quote_coverage"] >= 0.5
    assert report["depth_coverage"] >= 0.5


def test_assess_backtest_health_warns_when_ny_has_no_candidates_and_trades(monkeypatch):
    monkeypatch.delenv("BACKTEST_HEALTH_MIN_CANDIDATES", raising=False)
    monkeypatch.delenv("BACKTEST_HEALTH_MIN_TRADES", raising=False)
    monkeypatch.delenv("BACKTEST_HEALTH_MAX_DRAWDOWN_PCT", raising=False)

    status, reasons = _assess_backtest_health(
        strategy_name="ny_structure",
        rows=1000,
        candidates=0,
        entered_candidates=0,
        matrix_rows=0,
        trades=0,
        drawdown_pct=0.5,
    )

    assert status == "warning"
    assert any("low-candidates" in reason for reason in reasons)
    assert any("low-trades" in reason for reason in reasons)


def test_assess_backtest_health_warns_on_drawdown_breach(monkeypatch):
    monkeypatch.setenv("BACKTEST_HEALTH_MIN_CANDIDATES", "0")
    monkeypatch.setenv("BACKTEST_HEALTH_MIN_TRADES", "0")
    monkeypatch.setenv("BACKTEST_HEALTH_MAX_DRAWDOWN_PCT", "1.5")

    status, reasons = _assess_backtest_health(
        strategy_name="oneshot",
        rows=500,
        candidates=2,
        entered_candidates=1,
        matrix_rows=2,
        trades=1,
        drawdown_pct=2.0,
    )

    assert status == "warning"
    assert any("drawdown-breach" in reason for reason in reasons)
