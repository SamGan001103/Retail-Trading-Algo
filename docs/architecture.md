# Architecture

## Layers

- `config`: environment/config loading and validation
- `api`: authenticated ProjectX HTTP client
- `execution`: order/position control and bracket placement
- `runtime`: realtime subscriptions and bot loop
- `risk`: reusable risk guardrails
- `strategy`: signal interfaces
- `telemetry`: logging hooks

## Flow

1. Load runtime config from `.env`
2. Resolve contract id via API
3. Start realtime subscriptions
4. Run execution state loop
5. Enforce bracket/risk safety, flatten on fault

## Entrypoints

- `scripts/execution/start_trading.py`: master launcher with `--mode forward|backtest|train`
- `scripts/debug/`: operational debug scripts (lookups, order/position checks, flatten)
