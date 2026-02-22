# Architecture

## Layer Overview

1. `config`: env parsing, typed settings, symbol defaults
2. `api`: authenticated broker HTTP client and contract helpers
3. `broker`: provider adapter boundary (`projectx` implementation)
4. `runtime`: mode orchestration and forward realtime loop
5. `strategy`: strategy protocol + concrete strategies
6. `backtest`: historical data loaders + simulation engine
7. `position_management`: stop-loss/take-profit planners + limits
8. `ml`: training and runtime gate scaffolding
9. `telemetry`: structured telemetry sinks

## Active Runtime Paths

### Forward (`--mode forward`)

1. CLI (`scripts/execution/start_trading.py`) parses mode/strategy/profile.
2. `src/trading_algo/runtime/mode_runner.py` resolves config and strategy.
3. `src/trading_algo/runtime/bot_runtime.py` starts stream and executes strategy loop.
4. Strategy decisions place/cancel/flatten through broker adapter.
5. Telemetry appends to JSONL outputs.

### Backtest (`--mode backtest`)

1. `mode_runner` requires parquet orderflow input.
2. `backtest/data.py` streams parquet batches as `OrderFlowTick` events.
3. `backtest/engine.py` runs the event-driven simulator with rolling orderflow state.
4. Backtest telemetry writes parquet datasets (candidates, matrix, summary).

### Train (`--mode train`)

1. Load parquet dataset.
2. Build feature/label arrays.
3. Train model and persist artifact.

## Data Ingestion Paths

1. DBN -> parquet conversion: `scripts/data/convert_databento_dbn_to_parquet.py`
2. Parquet schema/sample inspection: `scripts/data/inspect_parquet_columns.py`

## Key Entrypoints

1. `scripts/execution/start_trading.py`
2. `src/trading_algo/runtime/mode_runner.py`
3. `src/trading_algo/runtime/bot_runtime.py`
4. `src/trading_algo/backtest/data.py`
5. `src/trading_algo/backtest/engine.py`

## Cleanup Notes

1. Forward runtime is strategy-only; legacy non-strategy execution loop was removed.
2. Backtest is parquet-only and streams batches to keep memory stable on large datasets.
