from datetime import datetime, timezone
from pathlib import Path

from trading_algo.backtest import OrderFlowTick
from trading_algo.runtime.mode_runner import (
    _BacktestMatrixTracker,
    _BacktestMatrixCsvWriter,
    _BacktestCandidateCsvWriter,
    _apply_shadow_ml,
    _build_backtest_scenarios,
    _build_backtest_shadow_gate,
    _build_walk_forward_windows,
    _ensure_orderflow_backtest_dataset,
    _load_major_news_blackouts,
    _latest_months_window,
    _latest_months_window_ticks,
    _validate_orderflow_backtest_preflight,
)
from trading_algo.backtest import BacktestConfig
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


def test_backtest_candidate_csv_writer_appends_rows(tmp_path: Path):
    csv_path = tmp_path / "backtest_candidates.csv"
    writer = _BacktestCandidateCsvWriter(
        str(csv_path),
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

    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("run_id,strategy,scenario_id,window_id,window_start_utc,window_end_utc,window_months")
    assert "cand-1" in lines[1]
    assert "cand-2" in lines[2]


def test_backtest_candidate_csv_writer_rotates_legacy_header(tmp_path: Path):
    csv_path = tmp_path / "backtest_candidates.csv"
    csv_path.write_text(
        "run_id,strategy,window_months,event_name\nold-run,ny_structure,6,strategy_candidate\n",
        encoding="utf-8",
    )
    writer = _BacktestCandidateCsvWriter(
        str(csv_path),
        run_id="run-2",
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
            "bar_index": 1,
            "bar_ts": "2026-01-01T00:01:00Z",
            "candidate_id": "cand-new",
            "status": "detected",
            "reason": "setup-ready",
            "side": "buy",
            "setup_index": 1,
            "setup_ts": "2026-01-01T00:01:00Z",
            "confluence_score": 2,
        }
    )
    rotated = list(tmp_path.glob("backtest_candidates.legacy_*.csv"))
    assert len(rotated) == 1
    out_lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert out_lines[0].startswith("run_id,strategy,scenario_id,window_id,window_start_utc,window_end_utc,window_months")
    assert "cand-new" in out_lines[1]


def test_backtest_matrix_csv_writer_rotates_legacy_header(tmp_path: Path):
    csv_path = tmp_path / "backtest_candidate_matrix.csv"
    csv_path.write_text(
        "run_id,strategy,window_months,candidate_id\nold-run,ny_structure,6,cand-legacy\n",
        encoding="utf-8",
    )
    writer = _BacktestMatrixCsvWriter(str(csv_path))
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
    rotated = list(tmp_path.glob("backtest_candidate_matrix.legacy_*.csv"))
    assert len(rotated) == 1
    out_lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert out_lines[0].startswith("run_id,strategy,scenario_id,window_id,window_start_utc,window_end_utc,window_months")
    assert "cand-new" in out_lines[1]


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
        _ensure_orderflow_backtest_dataset(ticks, "x.csv")
        assert False, "expected RuntimeError for missing depth fields"
    except RuntimeError as exc:
        assert "No usable depth data" in str(exc)


def test_load_major_news_blackouts_filters_major_and_merges(tmp_path: Path):
    news_path = tmp_path / "news.csv"
    news_path.write_text(
        "timestamp,impact,currency,event\n"
        "2026-01-01T14:30:00Z,high,USD,ISM\n"
        "2026-01-01T14:40:00Z,low,USD,minor\n"
        "2026-01-01T14:45:00Z,3,USD,NFP\n",
        encoding="utf-8",
    )
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


def test_validate_orderflow_preflight_report(tmp_path: Path, monkeypatch):
    csv_path = tmp_path / "ticks.csv"
    csv_path.write_text(
        "timestamp,seq,bid,ask,price,size,bestBidSize,bestAskSize\n"
        "2026-01-01T14:30:00Z,1,100.0,100.25,100.1,1,10,12\n"
        "2026-01-01T14:30:01Z,2,100.0,100.25,100.15,1,8,9\n",
        encoding="utf-8",
    )
    ticks = [
        OrderFlowTick(
            ts="2026-01-01T14:30:00Z",
            price=100.1,
            volume=1.0,
            quote={"bid": 100.0, "ask": 100.25},
            trade={"price": 100.1, "size": 1.0},
            depth={"bestBid": 100.0, "bestAsk": 100.25, "bestBidSize": 10.0, "bestAskSize": 12.0},
            seq=1,
        ),
        OrderFlowTick(
            ts="2026-01-01T14:30:01Z",
            price=100.15,
            volume=1.0,
            quote={"bid": 100.0, "ask": 100.25},
            trade={"price": 100.15, "size": 1.0},
            depth={"bestBid": 100.0, "bestAsk": 100.25, "bestBidSize": 8.0, "bestAskSize": 9.0},
            seq=2,
        ),
    ]
    monkeypatch.setenv("BACKTEST_PREFLIGHT_STRICT", "true")
    monkeypatch.setenv("BACKTEST_PREFLIGHT_MIN_ROWS", "1")
    monkeypatch.setenv("BACKTEST_PREFLIGHT_MIN_SESSION_ROWS", "1")
    monkeypatch.setenv("BACKTEST_PREFLIGHT_MIN_QUOTE_COVERAGE", "0.5")
    monkeypatch.setenv("BACKTEST_PREFLIGHT_MIN_DEPTH_COVERAGE", "0.5")
    monkeypatch.setenv("STRAT_AVOID_NEWS", "false")
    report = _validate_orderflow_backtest_preflight(
        data_csv=str(csv_path),
        ticks=ticks,
        window_start=datetime(2026, 1, 1, 14, 0, tzinfo=timezone.utc),
        window_end=datetime(2026, 1, 1, 15, 0, tzinfo=timezone.utc),
        news_blackouts=[],
    )
    assert report["rows"] == 2
    assert report["quote_coverage"] >= 0.5
    assert report["depth_coverage"] >= 0.5
