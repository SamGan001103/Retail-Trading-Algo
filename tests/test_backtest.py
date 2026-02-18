from pathlib import Path

from trading_algo.backtest import BacktestConfig, load_bars_from_csv, run_backtest
from trading_algo.strategy import OneShotLongStrategy


def test_load_bars_from_csv(tmp_path: Path):
    csv_path = tmp_path / "bars.csv"
    csv_path.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2026-01-01T00:00:00Z,100,101,99,100.5,1000\n"
        "2026-01-01T00:01:00Z,100.5,101.5,100,101,1100\n",
        encoding="utf-8",
    )
    bars = load_bars_from_csv(str(csv_path))
    assert len(bars) == 2
    assert bars[0].close == 100.5


def test_run_backtest_oneshot(tmp_path: Path):
    csv_path = tmp_path / "bars.csv"
    csv_path.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2026-01-01T00:00:00Z,100,101,99,100.0,1000\n"
        "2026-01-01T00:01:00Z,100,101,99,101.0,1000\n"
        "2026-01-01T00:02:00Z,101,102,100,102.0,1000\n"
        "2026-01-01T00:03:00Z,102,103,101,103.0,1000\n"
        "2026-01-01T00:04:00Z,103,104,102,104.0,1000\n"
        "2026-01-01T00:05:00Z,104,105,103,105.0,1000\n",
        encoding="utf-8",
    )
    bars = load_bars_from_csv(str(csv_path))
    strategy = OneShotLongStrategy(hold_bars=2, size=1)
    result = run_backtest(bars, strategy, BacktestConfig(initial_cash=10_000, fee_per_order=0, slippage_bps=0))
    assert result.num_trades >= 1
    assert result.final_equity > 0
