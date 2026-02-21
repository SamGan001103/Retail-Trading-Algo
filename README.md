# Retail Trading Algo (ProjectX / TopstepX)

This repository runs a futures trading stack with three modes:

1. `forward`: realtime broker-connected execution loop
2. `backtest`: historical simulation
3. `train`: ML model training scaffold (XGBoost)

Primary entrypoint:

```bash
python scripts/execution/start_trading.py
```

Supplemental docs:

1. `docs/architecture.md`
2. `docs/roadmap.md`

## 0. Progress Update (2026-02-21)

Current status for Databento backtesting pipeline:

1. End-to-end DBN day replay is working with local downloaded Databento zip data (`.dbn.zst`) using Python 3.11 + `databento` SDK.
2. Daily batch runner supports both DBN and CSV inputs and processes one day at a time with automatic temp cleanup.
3. NY strategy backtest path is strategy-only (no ML gate blocking), uses orderflow/tick replay, and strategy-planned SL/TP in execution replay.
4. Backtest outputs now include:
   - candidate lifecycle CSV (`artifacts/telemetry/backtest_candidates.csv`)
   - candidate matrix CSV (`artifacts/telemetry/backtest_candidate_matrix.csv`)
   - performance summary CSV (`artifacts/telemetry/backtest_summary.csv`)
5. Verified run example (single day):
   - `py -3.11 scripts/data/backtest_databento_daily.py --input "C:\Users\User\Downloads\GLBX-20260220-GSMJP896QR.zip" --strategy ny_structure --hold-bars 120 --profile normal --start-day 20260215 --end-day 20260215 --max-days 1 --continue-on-error`

## 1. User Instructions

### 1.1 Setup

Prerequisite: Python `>=3.11` (see `pyproject.toml`).

```bash
pip install -r requirements.txt
pip install -e .
```

Windows venv:

```bash
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

Optional train dependency (`--mode train`):

```bash
pip install xgboost
```

Optional Databento DBN import dependency (`mbp-10` batch downloads):

```bash
pip install databento-dbn
```

Optional Databento Historical SDK dependency (programmatic batch CSV download):

```bash
pip install databento
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
STRAT_FORWARD_BAR_SEC=60
STRAT_TICK_POLL_SEC=0.25
STRAT_TICK_POLL_IDLE_SEC=0.25
STRAT_TICK_POLL_ARMED_SEC=0.05
STRAT_DEPTH_WARNING_COOLDOWN_SEC=30
SUB_DEPTH=true

RUNTIME_PROFILE=normal
TELEMETRY_DIR=artifacts/telemetry
TELEMETRY_CANDIDATES_FILE=candidate_trades.jsonl
TELEMETRY_PERF_FILE=performance.jsonl
DEBUG_TICK_TRACE_EVERY_N=20

STRAT_NY_SESSION_START=09:30
STRAT_NY_SESSION_END=16:00
STRAT_TZ_NAME=America/New_York
STRAT_HTF_AGGREGATION=15
STRAT_BIAS_AGGREGATION=60
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
STRAT_ML_DECISION_POLICY=off
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

# Backtest candidate capture window/output.
BACKTEST_SLIP_ENTRY_TICKS=0.0
BACKTEST_SLIP_STOP_TICKS=0.0
BACKTEST_SLIP_TP_TICKS=0.0
BACKTEST_SPREAD_SLIP_K=1.0
BACKTEST_ENTRY_DELAY_EVENTS=1
BACKTEST_CANDIDATE_MONTHS=6
BACKTEST_WALK_FORWARD=false
BACKTEST_WF_WINDOW_MONTHS=6
BACKTEST_WF_STEP_MONTHS=1
# BACKTEST_WF_START_UTC=2025-01-01T00:00:00Z
# BACKTEST_WF_END_UTC=2026-01-01T00:00:00Z
BACKTEST_CANDIDATES_CSV=artifacts/telemetry/backtest_candidates.csv
BACKTEST_MATRIX_CSV=artifacts/telemetry/backtest_candidate_matrix.csv
BACKTEST_SUMMARY_CSV=artifacts/telemetry/backtest_summary.csv
BACKTEST_BAR_SEC=60
BACKTEST_SHADOW_ML_ENABLED=false
BACKTEST_SHADOW_ML_MODEL_PATH=artifacts/models/xgboost_model.json
BACKTEST_SHADOW_ML_MIN_PROBA=0.55
BACKTEST_PREFLIGHT_STRICT=true
BACKTEST_PREFLIGHT_REQUIRE_SEQ=true
BACKTEST_PREFLIGHT_MIN_ROWS=1
BACKTEST_PREFLIGHT_MIN_QUOTE_COVERAGE=0.90
BACKTEST_PREFLIGHT_MIN_DEPTH_COVERAGE=0.90
BACKTEST_PREFLIGHT_MIN_SESSION_ROWS=1
BACKTEST_SENSITIVITY_SWEEP=false
BACKTEST_SWEEP_ENTRY_DELAYS=0,1,2
BACKTEST_SWEEP_SLIP_ENTRY_TICKS=0,0.5,1.0
BACKTEST_SWEEP_SPREAD_SLIP_K=0,1,2

# Optional major-news avoidance during backtest.
STRAT_AVOID_NEWS=true
STRAT_NEWS_EXIT_ON_EVENT=false
BACKTEST_NEWS_CSV=data/news_usd_major.csv
BACKTEST_NEWS_PRE_MIN=15
BACKTEST_NEWS_POST_MIN=15
BACKTEST_NEWS_MAJOR_ONLY=true
BACKTEST_NEWS_CURRENCIES=USD

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
5. For `ny_structure` backtests, runtime forces `entry_mode=tick` and `require_orderflow=true`; each fill must pass tick-level orderflow gating.
6. Forward ML decision policy is controlled by `STRAT_ML_DECISION_POLICY`:
   - `off`: strategy-only decisions (no ML gate blocks entries)
   - `shadow`: evaluate ML score/reason but never block entries
   - `enforce`: gate entries with ML threshold checks
   If unset, runtime keeps backward compatibility (`enforce` when `STRAT_ML_GATE_ENABLED=true`, otherwise `off`).
7. `BACKTEST_SHADOW_ML_ENABLED=true` runs ML inference in backtest as log-only metadata (`ml_shadow_*`) without blocking entries.
8. `BACKTEST_CANDIDATE_MONTHS` selects the most recent window (default `6`) for non-walk-forward runs.
9. `BACKTEST_WALK_FORWARD=true` enables rolling windows (`BACKTEST_WF_WINDOW_MONTHS`, `BACKTEST_WF_STEP_MONTHS`) and tags rows with `window_id`.
10. `BACKTEST_CANDIDATES_CSV` appends candidate lifecycle events for each backtest run.
11. `BACKTEST_MATRIX_CSV` appends one ML-ready row per candidate with execution outcomes.
12. `BACKTEST_SUMMARY_CSV` appends strategy performance metrics per scenario/window (equity, pnl, return, win rate, drawdown).
13. `BACKTEST_PREFLIGHT_*` enforces dataset quality checks (required columns, UTC timestamps, `(timestamp,seq)` ordering, quote/depth/session coverage).
14. `BACKTEST_SLIP_*` and `BACKTEST_SPREAD_SLIP_K` control side-correct quote-based fills in tick replay.
15. `BACKTEST_ENTRY_DELAY_EVENTS` enforces next-event (or more) entry delay for tick entries.
16. `BACKTEST_SENSITIVITY_SWEEP=true` runs extra stress scenarios over latency/slippage/spread-slip settings.
17. `BACKTEST_BAR_SEC` controls bar aggregation during NY tick replay (default: `STRAT_FORWARD_BAR_SEC`, typically `60`).
18. `STRAT_AVOID_NEWS=true` plus `BACKTEST_NEWS_CSV` enables major-news blackout windows for setup/entry filtering.
19. `STRAT_TICK_POLL_SEC` is the base fallback poll interval; idle/armed values can override it.
20. Timeframe mapping uses `STRAT_FORWARD_BAR_SEC` as LTF base. Example shown above: LTF=1m, HTF=15m (`STRAT_HTF_AGGREGATION=15`), bias=1h (`STRAT_BIAS_AGGREGATION=60`).

### 1.3 Run Commands

CLI flags (`python scripts/execution/start_trading.py --help`):

```bash
--mode {forward,backtest,train}
--profile {normal,debug}
--data-csv DATA_CSV
--strategy STRATEGY
--model-out MODEL_OUT
--hold-bars HOLD_BARS
```

Accepted strategy aliases:

1. `oneshot`, `one_shot`, `one-shot`
2. `ny_structure`, `ny_session`, `market_structure`, `mnq_ny`

If `--data-csv` is omitted for `backtest` or `train`, runtime falls back to `BACKTEST_DATA_CSV` from env.

Forward:

```bash
python scripts/execution/start_trading.py --mode forward --strategy ny_structure --hold-bars 120
```

Forward strategy-only (recommended when model is not trained yet):

```bash
STRAT_ML_DECISION_POLICY=off STRAT_ML_GATE_ENABLED=false \
python scripts/execution/start_trading.py --mode forward --strategy ny_structure --hold-bars 120
```

Forward (`debug` profile for step-by-step strategy tracing):

```bash
python scripts/execution/start_trading.py --mode forward --profile debug --strategy ny_structure --hold-bars 120
```

Backtest (`ny_structure` requires orderflow-capable CSV):

```bash
python scripts/execution/start_trading.py --mode backtest --data-csv data/mnq_orderflow.csv --strategy ny_structure --hold-bars 120
```

Backtest walk-forward (example: 6m window, 1m step):

```bash
BACKTEST_WALK_FORWARD=true BACKTEST_WF_WINDOW_MONTHS=6 BACKTEST_WF_STEP_MONTHS=1 \
python scripts/execution/start_trading.py --mode backtest --data-csv data/mnq_orderflow.csv --strategy ny_structure --hold-bars 120
```

Backtest shadow-ML logging (no decision gating):

```bash
BACKTEST_SHADOW_ML_ENABLED=true BACKTEST_SHADOW_ML_MODEL_PATH=artifacts/models/xgboost_model.json \
python scripts/execution/start_trading.py --mode backtest --data-csv data/mnq_orderflow.csv --strategy ny_structure --hold-bars 120
```

ProjectX orderflow CSV capture (realtime stream -> backtest CSV):

```bash
python scripts/data/export_projectx_orderflow.py --symbol MNQ --duration-sec 1800 --output data/mnq_orderflow_capture.csv
```

Databento MBP-10 import (batch DBN/.zip -> backtest CSV):

```bash
python scripts/data/import_databento_orderflow.py --input "C:\Users\User\Downloads\GLBX-20260220-GSMJP896QR.zip" --output data/mnq_databento_orderflow.csv
```

Databento daily batch backtest from DBN (convert one day -> backtest -> delete temp CSV):

```bash
python scripts/data/backtest_databento_daily.py --input "C:\Users\User\Downloads\GLBX-20260220-GSMJP896QR.zip" --strategy ny_structure --hold-bars 120 --profile normal --continue-on-error
```

Databento daily batch backtest from CSV (no DBN decoder dependency):

```bash
python scripts/data/download_databento_batch_csv.py --start 2026-01-28T00:00:00 --end 2026-02-19T00:00:00 --symbols MNQ.c.0 --output-dir data/databento_daily_csv
python scripts/data/backtest_databento_daily.py --input "data/databento_daily_csv" --strategy ny_structure --hold-bars 120 --profile normal --continue-on-error
```

Databento daily batch backtest from pre-existing CSV directory:

```bash
python scripts/data/backtest_databento_daily.py --input "data/databento_daily_csv" --strategy ny_structure --hold-bars 120 --profile normal --continue-on-error
```

Notes:

1. This exporter captures realtime stream snapshots (quote/trade/depth), not vendor-side 6-month historical replay.
2. Set `SUB_DEPTH=true` so market depth is subscribed and written.
3. For `ny_structure` backtests keep depth enabled (default `--require-depth`).
4. For very large Databento archives, start with a subset: `--include "*20260218*"` or `--max-files 1`.
5. Daily batch runner supports DBN and CSV inputs, processes one day at a time, and frees local temp CSV storage by default.
6. `download_databento_batch_csv.py` can fetch split-by-day CSV batches directly from Databento (`DATABENTO_API_KEY` required).

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
6. Use `RUNTIME_PROFILE=normal` for cleaner output and lower runtime overhead.
7. Use `RUNTIME_PROFILE=debug` when you need strategy-process visualization and trace logs.

### 1.6 Verification

```bash
.\.venv\Scripts\python.exe -m pytest -q
```

Telemetry output files (JSONL append):

1. Candidate trades: `artifacts/telemetry/candidate_trades.jsonl`
2. Performance/execution: `artifacts/telemetry/performance.jsonl`
3. Backtest candidate CSV append: `artifacts/telemetry/backtest_candidates.csv` (configurable via `BACKTEST_CANDIDATES_CSV`)
4. Backtest ML matrix CSV append: `artifacts/telemetry/backtest_candidate_matrix.csv` (configurable via `BACKTEST_MATRIX_CSV`)
5. Backtest performance summary CSV append: `artifacts/telemetry/backtest_summary.csv` (configurable via `BACKTEST_SUMMARY_CSV`)
6. Backtest CSV rows now include `scenario_id` and `window_id` tags; candidate rows include `ml_shadow_*` fields when shadow mode is enabled.
7. If an existing backtest CSV has an older header schema, runtime rotates it to `*.legacy_<UTCSTAMP>.csv` and writes a fresh compatible header.
8. Forward execution events now include `candidate_id` when available, so candidate features can be joined to entry/exit outcomes for ML labeling.
9. Forward candidate events include structure/orderflow features such as `has_recent_sweep`, `htf_bias`, `bias_ok`, `continuation`, `reversal`, `equal_levels`, `fib_retracement`, `key_area_proximity`, `confluence_score`, `of_imbalance`, top-of-book sizes/prices, spread, and trade price/size.
10. Forward labels/outcomes come from execution events in `performance.jsonl` (`strategy_*_entry`, exits/flatten/protective exits) and can be joined on `candidate_id`.

### 1.7 Troubleshooting

If you see `ModuleNotFoundError: No module named 'trading_algo'`:

1. Re-run `pip install -e .` (or `.\.venv\Scripts\python.exe -m pip install -e .`).
2. Or set current shell path: `$env:PYTHONPATH="src"`.

## 2. Pipeline

### 2.1 Mode Pipeline

1. `forward`
   - Load runtime config
   - Resolve runtime profile (`normal`/`debug`) from `--profile` or `RUNTIME_PROFILE`
   - Initialize telemetry append sinks (candidate + performance JSONL)
   - Build broker adapter
   - Resolve contract
   - Start realtime streams
   - Run strategy loop (bar build + optional tick execution)
   - `debug` profile prints strategy-process traces; `normal` stays concise
2. `backtest`
   - `ny_structure`: load orderflow ticks/depth CSV, enforce tick sniper entry + orderflow gate, aggregate replay bars via `BACKTEST_BAR_SEC`
   - optional major-news blackout windows (`STRAT_AVOID_NEWS=true`, `BACKTEST_NEWS_CSV`)
   - other strategies: load OHLCV bars and run bar-only simulation
   - Simulate entries/exits with slippage/fees and bracket protections
   - Emit candidate/performance/execution telemetry and append candidate matrix rows for ML training
3. `train`
   - Build feature matrix from CSV
   - Train XGBoost model
   - Save model artifact

### 2.2 NY Structure Decision Pipeline

1. Module responsibilities
   - `swing_points.py`: stateful HTF/LTF swing detection and snapshots
   - `liquidity_sweep.py`: HTF aggregation, wick-sweep detection, sweep freshness
   - `structure_signals.py`: HTF bias, LTF trend, CHoCH, confluence checks, protocol-typed detector interfaces
   - `orderflow.py`: depth imbalance filtering, DOM liquidity levels, tick micro-timing/exhaustion
   - `ny_session_structure.py`: orchestration across setup, gating, sizing, and execution
2. Completed-bar setup environment
   - recent matching HTF sweep
   - HTF bias not opposing side
   - continuation or CHoCH reversal
   - confluence threshold met
3. Optional ML layer (`STRAT_ML_DECISION_POLICY`) can be off/shadow/enforce
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
   - NY orderflow replay executes/guards on ticks and still runs completed-bar strategy updates
   - optional major-news blackout can block setup/entry around economic events
   - Non-orderflow backtests simulate SL/TP brackets from bar high/low
   - if both hit in same bar, stop-loss is prioritized (conservative)
   - `ACCOUNT_MAX_DRAWDOWN_KILLSWITCH > 0` can halt new backtest entries via absolute drawdown threshold

## 3. Concerns and Plan

### 3.1 Current Concerns

1. Backtest fidelity now includes NY tick/depth replay, but quality still depends on historical depth coverage and CSV cleanliness.
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
| `scripts/data/export_projectx_orderflow.py` | ProjectX stream-to-CSV exporter. | Captures realtime quote/trade/depth snapshots into backtest orderflow CSV format. |
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
| `src/trading_algo/backtest/data.py` | Historical CSV parsing. | OHLCV `MarketBar` loader plus orderflow tick/depth loader. |
| `src/trading_algo/backtest/engine.py` | Backtest simulator. | Bar-based and tick-replay paths, slippage/fees, protective exits, drawdown halt, telemetry callback hooks. |
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
| `src/trading_algo/data_export/__init__.py` | Data export exports. | ProjectX capture API exports. |
| `src/trading_algo/data_export/projectx_orderflow.py` | ProjectX orderflow capture logic. | Stream polling, row normalization, depth/seq CSV writing, capture stats. |
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
| `src/trading_algo/runtime/bot_runtime.py` | Main forward runtime loop. | Stream polling, bar build, tick handling, order placement, guards, profile-aware runtime tracing, telemetry emission. |
| `src/trading_algo/runtime/drawdown_guard.py` | Runtime drawdown tracker/kill-switch helper. | Realized/unrealized PnL tracking and breach signaling. |
| `src/trading_algo/runtime/mode_runner.py` | Mode orchestrator. | Strategy factory, forward/backtest/train dispatch, news blackout loading, candidate/matrix CSV wiring. |
| `src/trading_algo/runtime/realtime_client.py` | Compatibility shim for legacy imports. | Alias to `ProjectXRealtimeStream`. |
| `src/trading_algo/strategy/__init__.py` | Strategy exports. | Base types + concrete strategy exports. |
| `src/trading_algo/strategy/base.py` | Strategy interfaces/types. | `StrategyDecision`, `MarketBar`, `PositionState`, protocol. |
| `src/trading_algo/strategy/simple.py` | Minimal sample strategy. | One-shot long strategy for baseline testing. |
| `src/trading_algo/strategy/market_structure/__init__.py` | Market-structure package exports. | Re-exports NY strategy plus sweep/swing/orderflow primitives. |
| `src/trading_algo/strategy/market_structure/liquidity_sweep.py` | HTF sweep utilities. | HTF bar aggregation, wick sweep detection, sweep freshness checks. |
| `src/trading_algo/strategy/market_structure/orderflow.py` | Orderflow/tick microstructure utilities. | Depth imbalance filter, DOM liquidity extraction, tick entry/exhaustion signals. |
| `src/trading_algo/strategy/market_structure/structure_signals.py` | Structure signal helpers. | LTF trend, HTF bias, CHoCH, equal-level/fib/key-area confluence checks, typed detector protocols. |
| `src/trading_algo/strategy/market_structure/swing_points.py` | Stateful swing-point detectors. | Swing level lifecycle, snapshots, multi-timeframe wrapper. |
| `src/trading_algo/strategy/market_structure/ny_session_structure.py` | Orchestrating NY session strategy. | Setup assembly, ML gate, SL/TP planning, risk sizing, bar/tick execution control. |
| `src/trading_algo/telemetry/__init__.py` | Telemetry exports. | Logger + telemetry router exports. |
| `src/trading_algo/telemetry/logging.py` | Logging helper. | Logger factory and shared logging defaults. |
| `src/trading_algo/telemetry/pipeline.py` | Telemetry router/sinks. | Profile-aware console trace + JSONL append pipeline for candidate/performance/execution events. |
| `tests/__init__.py` | Test package marker. | Enables package-style test imports. |
| `tests/test_backtest.py` | Backtest behavior tests. | CSV load, one-shot run, bracket simulation, drawdown halt tests. |
| `tests/test_backtest_orderflow.py` | Tick/depth replay tests. | Side-correct fills, `(timestamp,seq)` ordering, entry-delay semantics, and boundary handling checks. |
| `tests/test_bot_runtime_depth.py` | Forward runtime depth checks. | Depth payload availability detection for tick/orderflow runtime path. |
| `tests/test_broker_factory.py` | Broker factory tests. | Env-based adapter construction checks. |
| `tests/test_config.py` | Config/env parsing tests. | Runtime config loading and env parser checks. |
| `tests/test_drawdown_guard.py` | Drawdown guard tests. | Breach and realized-PnL accounting scenarios. |
| `tests/test_execution_engine.py` | Execution engine tests. | Bracket sign rules, payload checks, entry eligibility checks. |
| `tests/test_imports.py` | Import smoke tests. | Public module import validation. |
| `tests/test_ml_gate.py` | ML gate unit tests. | Disabled/fail-closed gate behavior checks. |
| `tests/test_mode_runner_backtest_candidates.py` | Backtest mode-runner data checks. | Windowing, preflight/news filters, candidate/matrix CSV wiring, and shadow-ML logging checks. |
| `tests/test_mode_runner_strategy_config.py` | Mode-runner strategy wiring tests. | Forward/backtest NY strategy entry/orderflow/ML-gate config behavior checks. |
| `tests/test_ny_session_structure_strategy.py` | NY strategy tests. | Session checks, setup arming, tick entry, ML rejection behavior, candidate-event emission checks. |
| `tests/test_orderflow_filter.py` | Orderflow filter tests. | Depth imbalance extraction and long/short gating checks. |
| `tests/test_projectx_orderflow_export.py` | ProjectX capture exporter tests. | Orderflow-row normalization, depth-required filtering, seq continuation checks. |
| `tests/test_position_management_planners.py` | SL/TP planner tests. | Stop planner, RRR filter, median target selection tests. |
| `tests/test_swing_points.py` | Swing detection tests. | Current/past swing transitions and swept-level behavior. |

## 5. Data Format (Backtest/Train)

### 5.1 OHLCV CSV (`train` + non-orderflow backtests)

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

### 5.2 Orderflow CSV (`ny_structure` backtest, required)

Minimum requirements:

1. timestamp column (`timestamp` / `datetime` / `time` / `date` / `ts_event` / `ts_recv`)
2. usable price (`trade_price` / `tradePrice` / `price` / `last` / `lastPrice` / `close` / `c`, or `bid` + `ask`)
3. usable depth on rows used for entry gating:
   - `bestBidSize` + `bestAskSize` (or aliases `bidSize`, `askSize`, `bid_size`, `ask_size`)
   - or JSON depth ladders (`depth_bids`, `depth_asks`)
4. deterministic order key required by default preflight: `seq` / `sequence` / `event_seq` (same-timestamp replay stability)

Useful optional columns:

1. quote prices: `bid`, `ask`
2. trade size: `trade_size`, `size`, `qty`, `quantity`, `lastSize`, `volume`, `v`
3. top-of-book prices: `bestBid`, `bestAsk`
4. event order key aliases: `seq`, `sequence`, `event_seq`, `eventSequence`

Databento MBP CSV compatibility:

1. Supports Databento-style timestamps in nanoseconds (`ts_event`, `ts_recv`) and normalizes to UTC ISO.
2. Supports top-of-book and depth ladder columns (`bid_px_00`..`bid_px_09`, `ask_px_00`..`ask_px_09`, `bid_sz_*`, `ask_sz_*`).
3. Supports Databento `sequence` for deterministic replay ordering.
4. Uses `action` to treat non-trade MBP updates as quote/depth-only rows (trade size stays `0` unless action is trade).

Databento batch downloads (`*.dbn.zst` in a `.zip`) are not consumed directly by runtime.
Convert them first:

```bash
python scripts/data/import_databento_orderflow.py --input "C:\Users\User\Downloads\GLBX-20260220-GSMJP896QR.zip" --output data/mnq_databento_orderflow.csv
```

Execution semantics in NY orderflow backtest:

1. market buy fills from ask; market sell fills from bid
2. long SL/TP triggers use bid side; short SL/TP triggers use ask side
3. quote-missing protective checks are ignored (conservative)

Example:

```csv
timestamp,seq,price,bid,ask,trade_size,bestBidSize,bestAskSize
2026-01-01T14:30:00Z,1001,21850.25,21850.00,21850.25,3,120,90
2026-01-01T14:30:00.250Z,1002,21850.50,21850.25,21850.50,2,132,86
```

### 5.3 News CSV (`ny_structure` backtest, optional)

Used only when `STRAT_AVOID_NEWS=true`.

Minimum columns:

1. event time: `timestamp` / `datetime` / `time` / `date`

Optional filters:

1. impact rank: `impact` / `importance` / `severity` / `priority` / `rank` / `impact_level`
2. currency tag: `currency` / `ccy` / `curr` / `country`

Major-event filtering:

1. with `BACKTEST_NEWS_MAJOR_ONLY=true`, values containing `high`, `major`, `red`, `critical`, or numeric `>=3` are treated as major
2. `BACKTEST_NEWS_CURRENCIES` (default `USD`) filters rows when a currency column exists
3. blackout intervals are evaluated in UTC with half-open semantics `[start, end)` (exact `end` timestamp is allowed)

Example:

```csv
timestamp,impact,currency,event
2026-01-10T13:30:00Z,high,USD,CPI
2026-01-15T13:30:00Z,3,USD,Retail Sales
```
