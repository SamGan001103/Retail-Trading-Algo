# Architecture

## Layers

- `config`: environment/config loading and validation
- `core`: shared constants/types used across runtime/strategy/backtest
- `broker`: broker adapter boundary (routing runtime calls to provider-specific implementations)
- `api`: authenticated ProjectX HTTP client
- `execution`: order/position control and bracket placement
- `runtime`: realtime subscriptions and bot loop
- `risk`: reusable risk guardrails
- `strategy`: signal interfaces
- `telemetry`: logging hooks

## Flow

1. Load runtime config from `.env`
2. Build broker adapter from config (`BROKER`, currently `projectx`)
3. Resolve contract id via adapter
4. Start realtime subscriptions
5. Run execution state loop
6. Enforce bracket/risk safety, flatten on fault

## Entrypoints

- `scripts/execution/start_trading.py`: master launcher with `--mode forward|backtest|train`
- `scripts/debug/`: operational debug scripts routed through broker adapter
