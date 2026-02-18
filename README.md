# Retail Trading Algo (ProjectX / TopstepX)

This repository is a Python execution and realtime foundation for automated trading on ProjectX/TopstepX.

## Structure

```text
src/
  trading_algo/
    config/
    api/
    execution/
    strategy/
    risk/
    runtime/
    telemetry/
scripts/
  execution/
  debug/
tests/
docs/
```

Canonical code lives under `src/trading_algo`.

## Current Capabilities

- Authenticated ProjectX REST client with token caching and connection reuse
- Realtime SignalR subscriptions for orders, positions, quotes, and trades
- Execution engine for market + bracket order placement, cancel, close, and flatten
- Runtime loop with basic safety checks (entry gating and bracket enforcement)

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

## Run

```bash
python scripts/execution/start_trading.py
```

Master execution aliases:

```bash
python scripts/execution/run_bot.py
python scripts/execution/bot.py
```

Debug and ops scripts:

```bash
python scripts/debug/account_lookup.py
python scripts/debug/account_check.py
python scripts/debug/market_lookup.py
python scripts/debug/order_place.py
python scripts/debug/order_cancel.py
python scripts/debug/orders_open.py
python scripts/debug/flatten_all.py
python scripts/debug/positions_open.py
python scripts/debug/position_close_contract.py
```

## Test

```bash
pytest -q
```

## Notes

- This is currently execution/state infrastructure, not a complete alpha strategy stack.
- Keep `.env` local and never commit live API secrets.
