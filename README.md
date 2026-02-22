# Retail Trading Algo

This repo runs three modes:

1. `forward`: realtime broker-connected execution
2. `backtest`: parquet-only historical orderflow replay
3. `train`: parquet-only ML training from backtest matrix outputs

Primary entrypoint:

```bash
python scripts/execution/start_trading.py
```

## Recommended Pipeline

1. Keep Databento DBN as raw immutable archive.
2. Convert DBN -> Parquet once.
3. Backtest by streaming parquet.
4. Train from backtest parquet telemetry outputs.

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

Windows venv:

```bash
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.\\.venv\\Scripts\\python.exe -m pip install -e .
```

Optional training dependency:

```bash
pip install xgboost
```

Optional DBN conversion dependency:

```bash
pip install databento
```

## Commands

Forward:

```bash
python scripts/execution/start_trading.py --mode forward --strategy ny_structure --hold-bars 120
```

Convert DBN -> Parquet (one-time):

```bash
python scripts/data/convert_databento_dbn_to_parquet.py --input "C:\\path\\GLBX-....zip" --output data/parquet/mbp10
python scripts/data/inspect_parquet_columns.py --input data/parquet/mbp10
```

Backtest from Parquet:

```bash
python scripts/execution/start_trading.py --mode backtest --data-parquet data/parquet/mbp10 --strategy ny_structure --hold-bars 120
```

Train from parquet matrix output:

```bash
python scripts/execution/start_trading.py --mode train --data-parquet artifacts/telemetry/backtest_candidate_matrix.parquet --model-out artifacts/models/xgboost_model.json
```

## Environment Keys (Backtest/Train)

- `BACKTEST_DATA_PARQUET`
- `BACKTEST_CANDIDATES_PARQUET`
- `BACKTEST_MATRIX_PARQUET`
- `BACKTEST_SUMMARY_PARQUET`
- `BACKTEST_PARQUET_BATCH_SIZE`
- `BACKTEST_NEWS_PATH` (optional)

## Outputs

Backtest parquet datasets:

1. `artifacts/telemetry/backtest_candidates.parquet`
2. `artifacts/telemetry/backtest_candidate_matrix.parquet`
3. `artifacts/telemetry/backtest_summary.parquet`

Forward telemetry:

1. `artifacts/telemetry/candidate_trades.jsonl`
2. `artifacts/telemetry/performance.jsonl`

## Core File Map

1. `scripts/execution/start_trading.py`: CLI mode launcher
2. `src/trading_algo/runtime/mode_runner.py`: mode orchestration
3. `src/trading_algo/runtime/bot_runtime.py`: forward loop
4. `src/trading_algo/backtest/data.py`: parquet orderflow loaders
5. `src/trading_algo/backtest/engine.py`: event-driven simulator
6. `src/trading_algo/strategy/market_structure/ny_session_structure.py`: primary strategy
7. `scripts/data/convert_databento_dbn_to_parquet.py`: DBN->Parquet conversion
8. `scripts/data/inspect_parquet_columns.py`: parquet schema/sample inspection

## Testing

```bash
.\\.venv\\Scripts\\python.exe -m pytest -q
```
