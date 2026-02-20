# Architecture

## Layers

- `config`: environment/config loading, validation, and symbol default profiles
- `core`: shared constants and side definitions
- `api`: authenticated ProjectX HTTP client and contract helpers
- `broker`: adapter boundary between runtime and provider-specific implementations
- `execution`: order placement, bracket signing, flatten/snapshot operations
- `strategy`: strategy protocol plus concrete strategies
- `position_management`: stop-loss/take-profit planners and position/order guards
- `runtime`: mode orchestration, realtime loop, drawdown guard, stream compatibility alias
- `backtest`: CSV loading and simulation engine (fees/slippage/bracket handling)
- `ml`: setup gate and training scaffold
- `telemetry`: logger helper hooks

## Runtime Flow (`forward`)

1. Load runtime config from `.env`.
2. Build broker adapter (`BROKER`, currently `projectx`).
3. Resolve contract id from symbol/live route.
4. Start realtime user/market streams.
5. Build bars from stream ticks.
6. Evaluate strategy decisions (bar and optional tick path).
7. Place market orders with bracket ticks from strategy decision or defaults.
8. Enforce runtime safeguards:
   - `position_management.guards` limits (positions/orders)
   - `runtime.drawdown_guard` kill-switch (halt + flatten on breach)
9. Continue until stopped.

## Backtest Flow (`backtest`)

1. Load OHLCV CSV into `MarketBar`.
2. Instantiate strategy through mode runner.
3. Simulate strategy decisions bar-by-bar.
4. Simulate bracket SL/TP hits from bar high/low.
5. Track equity, drawdown, trades, win rate, and returns.
6. Optionally halt new entries on absolute drawdown threshold.

## Train Flow (`train`)

1. Load labeled dataset from CSV.
2. Build simple feature/label arrays.
3. Train XGBoost model.
4. Persist model artifact for runtime ML gate.

## Entrypoints

- `scripts/execution/start_trading.py`: master launcher (`--mode forward|backtest|train`)
- `scripts/debug/`: operational broker-routed debug scripts
