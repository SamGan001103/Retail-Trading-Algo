# Retail Trading Algo (ProjectX / TopstepX)

This repository runs a futures trading stack with three modes:

1. `forward`: realtime broker-connected execution loop
2. `backtest`: historical simulation
3. `train`: ML model training scaffold (XGBoost)

Primary entrypoint:

```bash
python scripts/execution/start_trading.py
```

## 1. User Instructions

### 1.1 Setup

```bash
pip install -r requirements.txt
pip install -e .
```

Windows venv:

```bash
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

CLI scripts import `trading_algo` from the `src/` package path. Before running commands, use either:

1. Editable install: `pip install -e .`
2. Session PYTHONPATH override (PowerShell): `$env:PYTHONPATH="src"`

### 1.2 Configure `.env`

Use this as the minimum runtime template:

```env
BROKER=projectx
BROKER_BASE_URL=https://...
BROKER_USERNAME=...
BROKER_API_KEY=...
ACCOUNT_ID=123456
BROKER_USER_HUB_URL=https://.../hubs/user
BROKER_MARKET_HUB_URL=https://.../hubs/market

BOT_ENABLED=0
TRADING_ENVIRONMENT=DEMO
LIVE=false
FLATTEN_ON_START=false
TRADE_ON_START=false
EXIT_GRACE_SEC=5

SYMBOL=MNQ
SIDE=0
SIZE=1
SL_TICKS=40
TP_TICKS=80
LOOP_SEC=1
```

NY structure + position management knobs used by current code:

```env
STRAT_ENTRY_MODE=tick
STRAT_REQUIRE_ORDERFLOW=false
STRAT_FORWARD_BAR_SEC=1
STRAT_TICK_POLL_SEC=0.25
STRAT_TICK_POLL_IDLE_SEC=0.25
STRAT_TICK_POLL_ARMED_SEC=0.05
SUB_DEPTH=true

STRAT_NY_SESSION_START=09:30
STRAT_NY_SESSION_END=16:00
STRAT_TZ_NAME=America/New_York
STRAT_HTF_AGGREGATION=5
STRAT_HTF_SWING_HIGH=5
STRAT_HTF_SWING_LOW=5
STRAT_LTF_SWING_HIGH=3
STRAT_LTF_SWING_LOW=3
STRAT_SWEEP_WICK_MIN=0.5
STRAT_SWEEP_EXPIRY_BARS=40
STRAT_EQUAL_LEVEL_TOL_BPS=8
STRAT_KEY_AREA_TOL_BPS=12
STRAT_MIN_CONFLUENCE=1
STRAT_TICK_SETUP_EXPIRY_BARS=3
STRAT_TICK_HISTORY_SIZE=120
STRAT_TICK_MIN_IMBALANCE=0.12
STRAT_TICK_MIN_TRADE_SIZE=1.0
STRAT_TICK_SPOOF_COLLAPSE=0.35
STRAT_TICK_ABSORPTION_TRADES=2
STRAT_TICK_ICEBERG_RELOADS=2

STRAT_ML_GATE_ENABLED=false
STRAT_ML_MODEL_PATH=artifacts/models/xgboost_model.json
STRAT_ML_MIN_PROBA=0.55
STRAT_ML_FAIL_OPEN=false

# Split drawdown controls:
# - STRAT_ACCOUNT_MAX_DRAWDOWN: sizing budget anchor.
# - ACCOUNT_MAX_DRAWDOWN_KILLSWITCH: runtime/backtest trading halt threshold.
STRAT_ACCOUNT_MAX_DRAWDOWN=2000
ACCOUNT_MAX_DRAWDOWN_KILLSWITCH=2000
ACCOUNT_MAX_DRAWDOWN=2000

STRAT_MAX_TRADE_DRAWDOWN_FRACTION=0.15
STRAT_MIN_RRR=3.0
STRAT_MAX_RRR=10.0
STRAT_SL_NOISE_BUFFER_TICKS=2
STRAT_SL_MAX_TICKS=200
STRAT_TP_FRONT_RUN_TICKS=2
STRAT_ML_MIN_SIZE_FRACTION=0.35
STRAT_ML_SIZE_FLOOR_SCORE=0.55
STRAT_ML_SIZE_CEILING_SCORE=0.90
STRAT_ENABLE_EXHAUSTION_MARKET_EXIT=true
STRAT_DRAWDOWN_GUARD_ENABLED=true
STRAT_MAX_OPEN_POSITIONS=1
STRAT_MAX_OPEN_ORDERS_WHILE_FLAT=0

# Optional per-symbol overrides.
# If omitted, defaults are inferred from SYMBOL profile (MNQ/NQ/MES/ES/MGC/GC).
# STRAT_TICK_SIZE=0.25
# STRAT_TICK_VALUE=0.5
# STRAT_DOM_LIQUIDITY_WALL_SIZE=800
```

Notes:

1. `BROKER_*` keys are preferred; legacy aliases (`PROJECTX_*`, `RTC_*`) are still accepted.
2. `SIDE` accepts `0/1` and `buy/sell` or `long/short`.
3. Use `.env.example` as the baseline and keep real credentials only in local `.env`.
4. Mode-focused templates are available: `.env.forward.example`, `.env.backtest.example`, `.env.train.example`.
5. `STRAT_ENTRY_MODE` defaults to `tick` in `forward` mode and `bar` in `backtest` mode unless explicitly set.
6. `STRAT_TICK_POLL_SEC` is the base fallback poll interval; idle/armed values can override it.

### 1.3 Run Commands

CLI flags (`python scripts/execution/start_trading.py --help`):

```bash
--mode {forward,backtest,train}
--data-csv DATA_CSV
--strategy STRATEGY
--model-out MODEL_OUT
--hold-bars HOLD_BARS
```

Accepted strategy aliases:

1. `oneshot`, `one_shot`, `one-shot`
2. `ny_structure`, `ny_session`, `market_structure`, `mnq_ny`

Forward:

```bash
python scripts/execution/start_trading.py --mode forward --strategy ny_structure --hold-bars 120
```

Backtest:

```bash
python scripts/execution/start_trading.py --mode backtest --data-csv data/ohlcv.csv --strategy ny_structure --hold-bars 120
```

Train:

```bash
python scripts/execution/start_trading.py --mode train --data-csv data/ohlcv.csv --model-out artifacts/models/xgboost_model.json
```

### 1.4 Debug / Ops Commands

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

### 1.5 Safety Checklist

1. Keep `BOT_ENABLED=0` while validating setup.
2. Use `scripts/debug/account_check.py` to verify account routing.
3. Keep `LIVE=false` until forward testing sign-off.
4. Start with `SIZE=1`.
5. Confirm both `STRAT_ACCOUNT_MAX_DRAWDOWN` and `ACCOUNT_MAX_DRAWDOWN_KILLSWITCH` before enabling.

### 1.6 Verification

```bash
.\.venv\Scripts\python.exe -m pytest -q
```

## 2. Pipeline

### 2.1 Mode Pipeline

1. `forward`
   - Load runtime config
   - Build broker adapter
   - Resolve contract
   - Start realtime streams
   - Run strategy loop (bar build + optional tick execution)
2. `backtest`
   - Load CSV bars
   - Instantiate strategy
   - Simulate entries/exits with slippage/fees
   - Simulate bracket SL/TP hits from OHLC bars
3. `train`
   - Build feature matrix from CSV
   - Train XGBoost model
   - Save model artifact

### 2.2 NY Structure Decision Pipeline

1. Module responsibilities
   - `swing_points.py`: stateful HTF/LTF swing detection and snapshots
   - `liquidity_sweep.py`: HTF aggregation, wick-sweep detection, sweep freshness
   - `structure_signals.py`: HTF bias, LTF trend, CHoCH, confluence checks
   - `orderflow.py`: depth imbalance filtering, DOM liquidity levels, tick micro-timing/exhaustion
   - `ny_session_structure.py`: orchestration across setup, gating, sizing, and execution
2. Completed-bar setup environment
   - recent matching HTF sweep
   - HTF bias not opposing side
   - continuation or CHoCH reversal
   - confluence threshold met
3. ML gate scores setup and approves/rejects
4. Position management planning
   - stop-loss level/ticks
   - take-profit level/ticks
   - contract sizing with risk cap and ML scaling
5. Execution
   - `entry_mode=bar`: place immediately
   - `entry_mode=tick`: arm setup, wait for `tick_entry_ready(...)` micro-timing
6. Runtime safeguards
   - position/order count limits
   - drawdown guard (halt + flatten on breach)
   - optional in-position exhaustion market exit (`tick_exhaustion_exit_signal(...)`)

### 2.3 Risk and Execution Behavior

1. Per-trade risk budget:
   - `risk_budget = STRAT_ACCOUNT_MAX_DRAWDOWN * STRAT_MAX_TRADE_DRAWDOWN_FRACTION`
2. Contracts:
   - `max_contracts = floor(risk_budget / (stop_ticks * STRAT_TICK_VALUE))`
   - final size constrained by `SIZE` and ML scaling
3. RRR filter:
   - only trades with `STRAT_MIN_RRR <= RRR <= STRAT_MAX_RRR`
4. Stops/targets:
   - stop uses invalidation+noise buffer planning
   - target uses valid target levels and median choice when multiple qualify
5. Backtest specifics:
   - SL/TP brackets simulated from bar high/low
   - if both hit in same bar, stop-loss is prioritized (conservative)
   - `ACCOUNT_MAX_DRAWDOWN_KILLSWITCH > 0` can halt new backtest entries via absolute drawdown threshold

## 3. Concerns and Plan

### 3.1 Current Concerns

1. Realtime vs backtest fidelity is still bar-based for many orderflow concepts.
2. ML model lifecycle is scaffolded; production validation loops are not fully automated.
3. Regime drift can degrade both rule edge and ML score quality.
4. Runtime controls are improving but still need stronger monitoring/alerting surfaces.

### 3.2 Practical Plan

1. Data and labeling
   - log every setup candidate (accepted and rejected)
   - include structure context, ML score, orderflow features, realized outcome
2. Validation
   - walk-forward evaluation with fixed promotion criteria
   - measure stability by regime, not only global averages
3. Runtime hardening
   - expand kill-switches (session/daily caps)
   - add alerts for stream faults, stale data, execution anomalies
4. Deployment
   - micro-size forward stage
   - staged scale-up only after objective thresholds are met

## 4. File Tree Reference

This table reflects the current working tree files (excluding cache/temp folders).

| Path | What it does | Key contents |
| --- | --- | --- |
| `.env` | Local runtime configuration (not committed by default). | Broker creds, symbol, strategy/risk knobs, runtime flags. |
| `.env.example` | Safe baseline config template. | Placeholder values, split drawdown vars, optional override comments. |
| `.env.forward.example` | Forward-mode template. | Broker/runtime/strategy settings for live forward loop. |
| `.env.backtest.example` | Backtest-mode template. | Data/simulation/risk settings for historical runs. |
| `.env.train.example` | Train-mode template. | Dataset/model artifact defaults for training runs. |
| `.gitignore` | Git ignore rules. | `.env`, virtualenv, caches, build artifacts. |
| `.vscode/settings.json` | Editor/workspace settings. | VS Code local configuration. |
| `README.md` | Main operator documentation. | Setup, run instructions, pipeline, concerns, file map. |
| `conftest.py` | Pytest bootstrap config. | Test path/bootstrap helpers. |
| `pyproject.toml` | Build/tool configuration. | Packaging metadata, pyright/pytest/format settings. |
| `requirements.txt` | Runtime dependency list. | `python-dotenv`, `requests`, `signalrcore`, etc. |
| `setup.py` | Setuptools packaging entry. | `src`-layout package discovery/install metadata. |
| `docs/architecture.md` | Supplemental architecture notes. | Layering and high-level runtime flow. |
| `docs/roadmap.md` | Supplemental roadmap notes. | Delivery milestones and priorities. |
| `scripts/execution/start_trading.py` | Master CLI entrypoint. | Argument parsing and mode dispatch to runtime. |
| `scripts/debug/_common.py` | Shared helper for debug scripts. | Loads runtime config + broker adapter. |
| `scripts/debug/account_check.py` | Validate configured account exists/tradable. | Account lookup and target account assertion. |
| `scripts/debug/account_lookup.py` | Print available accounts. | Account listing and selected fields dump. |
| `scripts/debug/flatten_all.py` | Emergency flatten helper. | Cancel open orders + close open positions. |
| `scripts/debug/market_lookup.py` | Contract search helper. | Symbol/contract lookup through adapter. |
| `scripts/debug/order_place.py` | Manual test order placement. | Sends market order with bracket params. |
| `scripts/debug/order_cancel.py` | Manual order cancel helper. | Cancel by order id through adapter. |
| `scripts/debug/orders_open.py` | Inspect open orders. | Prints currently open orders. |
| `scripts/debug/positions_open.py` | Inspect open positions. | Prints currently open positions. |
| `scripts/debug/position_close_contract.py` | Close a specific contract position. | Contract-targeted close request. |
| `src/trading_algo/__init__.py` | Package root exports. | Module namespace list (`api`, `runtime`, `position_management`, etc.). |
| `src/trading_algo/api/__init__.py` | API layer exports. | ProjectX client + contract helpers. |
| `src/trading_algo/api/client.py` | Authenticated HTTP API client. | Token handling, POST wrapper, request logging hooks. |
| `src/trading_algo/api/contracts.py` | Contract search/resolution helpers. | Search contracts and resolve single contract id. |
| `src/trading_algo/api/factory.py` | API client factory from env. | Env loading and `ProjectXClient` construction. |
| `src/trading_algo/backtest/__init__.py` | Backtest exports. | CSV loader + backtest engine symbols. |
| `src/trading_algo/backtest/data.py` | Historical CSV parsing. | Column alias handling and `MarketBar` conversion. |
| `src/trading_algo/backtest/engine.py` | Backtest simulator. | Slippage/fees, bracket SL/TP checks, drawdown halt, metrics. |
| `src/trading_algo/broker/__init__.py` | Broker layer exports. | Adapter interfaces and ProjectX adapter exports. |
| `src/trading_algo/broker/base.py` | Broker protocol contracts. | `BrokerAdapter` and stream protocol signatures. |
| `src/trading_algo/broker/factory.py` | Broker adapter factory. | Runtime config -> concrete adapter selection. |
| `src/trading_algo/broker/projectx.py` | ProjectX adapter implementation. | Resolve contracts, stream creation, execution operations. |
| `src/trading_algo/broker/projectx_realtime.py` | ProjectX SignalR realtime stream. | Quote/trade/depth subscriptions, state cache, reconnect flow. |
| `src/trading_algo/config/__init__.py` | Config exports. | `RuntimeConfig`, env parsers, config loader. |
| `src/trading_algo/config/env.py` | Primitive env parsing helpers. | `env_bool`, `env_int`, `env_float`, required env helper. |
| `src/trading_algo/config/settings.py` | Runtime configuration model. | `RuntimeConfig` dataclass + loader/validation. |
| `src/trading_algo/config/symbol_profile.py` | Symbol default profile mapping. | Tick size/value and DOM wall defaults by symbol. |
| `src/trading_algo/core/__init__.py` | Core exports. | Side constants and aliases. |
| `src/trading_algo/core/side.py` | Shared side constants. | `BUY`/`SELL` and related typing helpers. |
| `src/trading_algo/execution/__init__.py` | Execution exports. | Engine and bracket signing exports. |
| `src/trading_algo/execution/engine.py` | Order/position execution engine. | Place market+brackets, snapshot, flatten, safety checks. |
| `src/trading_algo/execution/factory.py` | Execution engine from env. | Build API client + `ExecutionEngine` pair. |
| `src/trading_algo/ml/__init__.py` | ML exports. | Gate and trainer symbols. |
| `src/trading_algo/ml/gate.py` | Setup ML gate logic. | Setup feature mapping, model load, approve/reject decision. |
| `src/trading_algo/ml/trainer.py` | XGBoost training scaffold. | Feature/label extraction from CSV and model save. |
| `src/trading_algo/position_management/__init__.py` | Position management exports. | Guards + SL/TP planner exports. |
| `src/trading_algo/position_management/guards.py` | Runtime position/order guardrails. | `RiskLimits`, `enforce_position_limits`. |
| `src/trading_algo/position_management/stop_loss.py` | Stop-loss planner. | Invalidation-level stop selection and noise buffer logic. |
| `src/trading_algo/position_management/take_profit.py` | Take-profit planner. | RRR-filtered target selection and median valid target logic. |
| `src/trading_algo/runtime/__init__.py` | Runtime exports. | Mode runner, main runtime entry, drawdown guard exports. |
| `src/trading_algo/runtime/bot_runtime.py` | Main forward runtime loop. | Stream polling, bar build, tick handling, order placement, guards. |
| `src/trading_algo/runtime/drawdown_guard.py` | Runtime drawdown tracker/kill-switch helper. | Realized/unrealized PnL tracking and breach signaling. |
| `src/trading_algo/runtime/mode_runner.py` | Mode orchestrator. | Strategy factory, forward/backtest/train dispatch, env wiring. |
| `src/trading_algo/runtime/realtime_client.py` | Compatibility shim for legacy imports. | Alias to `ProjectXRealtimeStream`. |
| `src/trading_algo/strategy/__init__.py` | Strategy exports. | Base types + concrete strategy exports. |
| `src/trading_algo/strategy/base.py` | Strategy interfaces/types. | `StrategyDecision`, `MarketBar`, `PositionState`, protocol. |
| `src/trading_algo/strategy/simple.py` | Minimal sample strategy. | One-shot long strategy for baseline testing. |
| `src/trading_algo/strategy/market_structure/__init__.py` | Market-structure package exports. | Re-exports NY strategy plus sweep/swing/orderflow primitives. |
| `src/trading_algo/strategy/market_structure/liquidity_sweep.py` | HTF sweep utilities. | HTF bar aggregation, wick sweep detection, sweep freshness checks. |
| `src/trading_algo/strategy/market_structure/orderflow.py` | Orderflow/tick microstructure utilities. | Depth imbalance filter, DOM liquidity extraction, tick entry/exhaustion signals. |
| `src/trading_algo/strategy/market_structure/structure_signals.py` | Structure signal helpers. | LTF trend, HTF bias, CHoCH, equal-level/fib/key-area confluence checks. |
| `src/trading_algo/strategy/market_structure/swing_points.py` | Stateful swing-point detectors. | Swing level lifecycle, snapshots, multi-timeframe wrapper. |
| `src/trading_algo/strategy/market_structure/ny_session_structure.py` | Orchestrating NY session strategy. | Setup assembly, ML gate, SL/TP planning, risk sizing, bar/tick execution control. |
| `src/trading_algo/telemetry/__init__.py` | Telemetry exports. | Logger helper exports. |
| `src/trading_algo/telemetry/logging.py` | Logging helper. | Logger factory and shared logging defaults. |
| `tests/__init__.py` | Test package marker. | Enables package-style test imports. |
| `tests/test_backtest.py` | Backtest behavior tests. | CSV load, one-shot run, bracket simulation, drawdown halt tests. |
| `tests/test_broker_factory.py` | Broker factory tests. | Env-based adapter construction checks. |
| `tests/test_config.py` | Config/env parsing tests. | Runtime config loading and env parser checks. |
| `tests/test_drawdown_guard.py` | Drawdown guard tests. | Breach and realized-PnL accounting scenarios. |
| `tests/test_execution_engine.py` | Execution engine tests. | Bracket sign rules, payload checks, entry eligibility checks. |
| `tests/test_imports.py` | Import smoke tests. | Public module import validation. |
| `tests/test_ml_gate.py` | ML gate unit tests. | Disabled/fail-closed gate behavior checks. |
| `tests/test_ny_session_structure_strategy.py` | NY strategy tests. | Session checks, setup arming, tick entry, ML rejection behavior. |
| `tests/test_orderflow_filter.py` | Orderflow filter tests. | Depth imbalance extraction and long/short gating checks. |
| `tests/test_position_management_planners.py` | SL/TP planner tests. | Stop planner, RRR filter, median target selection tests. |
| `tests/test_swing_points.py` | Swing detection tests. | Current/past swing transitions and swept-level behavior. |

## 5. Data Format (Backtest/Train)

Accepted OHLCV column aliases:

1. timestamp: `timestamp`, `datetime`, `time`, `date`
2. open: `open`, `o`
3. high: `high`, `h`
4. low: `low`, `l`
5. close: `close`, `c`
6. volume: `volume`, `v`

Example:

```csv
timestamp,open,high,low,close,volume
2026-01-01T00:00:00Z,100,101,99,100.5,1000
2026-01-01T00:01:00Z,100.5,101.2,100.1,100.9,1200
```
