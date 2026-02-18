# Retail Trading Algo (ProjectX / TopstepX)

This README is the user instruction manual for running the algo in:

- `forward` mode (realtime/forward testing, broker-connected)
- `backtest` mode (historical simulation)
- `train` mode (ML training scaffold for XGBoost)

## 1. Project Layout

```text
src/
  trading_algo/          # Core implementation
    broker/              # Broker adapter boundary (current implementation: projectx)
    core/                # Shared constants/types (broker-agnostic)
scripts/
  execution/             # Main launchers
  debug/                 # Debug/ops utilities (routed through broker adapter)
tests/
docs/
```

Canonical code is in `src/trading_algo`.

## 2. Setup

From repo root:

```bash
pip install -r requirements.txt
pip install -e .
```

If using venv on Windows:

```bash
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

## 3. One Main File To Start

Run this file for all modes:

```bash
python scripts/execution/start_trading.py
```

Select mode with `--mode`:

- `--mode forward`
- `--mode backtest`
- `--mode train`

## 4. Environment Variables (`.env`)

### Required for `forward` mode

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

SYMBOL=MNQ
LIVE=false
SIDE=0
SIZE=1
SL_TICKS=40
TP_TICKS=80
LOOP_SEC=1
EXIT_GRACE_SEC=5
FLATTEN_ON_START=false
TRADE_ON_START=false
```

Notes:

- `BROKER` defaults to `projectx` if omitted.
- `SIDE` accepts `0/1` and also `buy/sell` or `long/short`.
- Preferred broker-neutral keys are `BROKER_*` and are used by runtime/config.
- Current adapter implementation is `projectx`; broker abstraction is in `src/trading_algo/broker/`.
- Legacy ProjectX keys are still accepted for backward compatibility:
  - `PROJECTX_BASE_URL`, `PROJECTX_USERNAME`, `PROJECTX_API_KEY`
  - `RTC_USER_HUB_URL`, `RTC_MARKET_HUB_URL`

### Optional for `backtest` mode

```env
BACKTEST_DATA_CSV=data/ohlcv.csv
BACKTEST_INITIAL_CASH=10000
BACKTEST_FEE_PER_ORDER=1.0
BACKTEST_SLIPPAGE_BPS=1.0
```

## 5. Run Commands

### A) Forward testing / realtime

```bash
python scripts/execution/start_trading.py --mode forward
```

Important:

- `BOT_ENABLED=1` is required to actually run trading loop.
- Keep `LIVE=false` unless you intentionally switch to live routing.
- Forward runtime fails fast if realtime user/market streams do not connect at startup.

### B) Backtesting

```bash
python scripts/execution/start_trading.py --mode backtest --data-csv data/ohlcv.csv --strategy oneshot --hold-bars 20
```

You can also omit `--data-csv` if `BACKTEST_DATA_CSV` is set in `.env`.

### C) ML training scaffold (XGBoost)

```bash
python scripts/execution/start_trading.py --mode train --data-csv data/ohlcv.csv --model-out artifacts/models/xgboost_model.json
```

If `xgboost` is missing:

```bash
pip install xgboost
```

## 6. Historical CSV Format (Backtest/Train)

CSV must include OHLCV columns. Accepted names:

- timestamp: `timestamp` or `datetime` or `time` or `date`
- open: `open` or `o`
- high: `high` or `h`
- low: `low` or `l`
- close: `close` or `c`
- volume: `volume` or `v`

Example:

```csv
timestamp,open,high,low,close,volume
2026-01-01T00:00:00Z,100,101,99,100.5,1000
2026-01-01T00:01:00Z,100.5,101.2,100.1,100.9,1200
```

## 7. Debug/Ops Scripts

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

All debug scripts call the broker adapter interface, so the same commands are reusable across supported broker adapters.

## 8. Quick Safety Checklist

Before enabling trading:

1. Keep `BOT_ENABLED=0` while validating connections.
2. Confirm account with `scripts/debug/account_check.py`.
3. Confirm market contract with `scripts/debug/market_lookup.py`.
4. Keep `LIVE=false` for forward testing.
5. Use small size (`SIZE=1`) initially.

## 9. Tests

```bash
pytest -q
```

Windows venv:

```bash
.\.venv\Scripts\python.exe -m pytest -q
```

## 10. Current Scope

- Good for infrastructure validation and forward testing.
- Backtest/train are implemented as scaffolding and can be extended.
- Strategy and risk logic are not final production alpha yet.

## 11. Main Entry Summary

If you remember only one file, use:

```bash
scripts/execution/start_trading.py
```

Then choose mode with `--mode`.

## 12. Build Pipeline Checklist (Start -> Finish)

Use this as the master progress tracker.

- [x] Step 1: Repository structure organized (`src/`, `scripts/execution`, `scripts/debug`, `tests`, `docs`).
- [x] Step 2: Core broker execution foundation in place (auth client, execution engine, realtime runtime loop).
- [x] Step 3: Master mode switch in place (`forward`, `backtest`, `train`) via `scripts/execution/start_trading.py`.
- [x] Step 4: Backtest scaffold implemented (CSV loader, basic simulator, PnL/return/win-rate/max-drawdown metrics).
- [x] Step 5: ML training scaffold implemented (`train` mode + XGBoost trainer skeleton).
- [ ] Step 6: Turn your trading strategy into code (clear entry/exit, invalidation, sizing, and session rules).
- [ ] Step 7: Download and curate historical market data (clean OHLCV, session boundaries, symbol rollover handling).
- [ ] Step 8: Backtest the coded strategy and log every trade candidate/outcome (market structure snapshot + result).
- [ ] Step 9: Build a strategy-trade dataset from those logs (one row per candidate trade, with labels).
- [ ] Step 10: Feature engineer the collected trade dataset (no leakage, consistent feature versions).
- [ ] Step 11: Train and validate ML classifier (XGBoost) on historical strategy trades.
- [ ] Step 12: Integrate ML gate into strategy runtime (strategy proposes trade, model approves/rejects).
- [ ] Step 13: Upgrade backtest realism and validation (fees/slippage, walk-forward, out-of-sample checks).
- [ ] Step 14: Harden risk and production controls (daily limits, cooldowns, kill-switch, monitoring and alerts).
- [ ] Step 15: Forward test sign-off and staged live rollout (micro-size first, rollback plan ready).
- [ ] Step 16: Continuous retraining loop (new trades appended, periodic retrain/re-validate/redeploy).

Current focus:

- Next practical milestone is Step 6 -> Step 8 (code strategy, get data, backtest and record trade outcomes).

## 13. Drift Concerns and Adaptive Plan (Working Notes)

Current system is a 2-layer decision process:

1. `Strategy layer (rule-based)`: finds candidate setups (market structure + price action + orderflow).
2. `ML layer`: classifies/regresses candidate quality to decide take/skip.

Known drift risks:

- `ML drift`: model quality decays as market distribution changes.
- `Rule drift`: setup logic itself stops matching current market behavior (frequency, quality, and execution response drift).

### A) ML Layer Drift Handling (baseline plan)

- Train on rolling windows (recent data weighted higher).
- Retrain on schedule (`daily` / `weekly` / `monthly`) based on data volume and strategy horizon.
- Use walk-forward validation and out-of-sample gates before promotion.

### B) Rule Layer Drift Handling (main concern)

Treat rule logic as adaptive, not fixed:

- Parameterize rule family instead of one hardcoded setup definition.
- Maintain a `champion + challengers` set of rule variants.
- Track per-variant health:
  - candidate frequency
  - win rate / expectancy
  - net PnL after costs
  - slippage/fill quality
  - max drawdown / instability
- Add drift/change detectors (example methods: ADWIN/CUSUM/changepoint).
- On drift trigger:
  - reduce or pause variant weight
  - run local re-optimization on recent window
  - promote challenger only after out-of-sample check

### C) Regime-Aware Adaptation

Use regime features (volatility, trend strength, liquidity/orderflow state) to route decisions:

- Map market state -> preferred rule variant set.
- Keep per-regime performance stats.
- Reweight variants online with strict risk caps.

### D) Guardrails (must stay static)

Adaptation is allowed only inside predefined limits:

- hard position/risk caps
- kill-switch on abnormal drawdown or execution degradation
- cooldown rules after loss clusters
- rollback path to last stable champion

### E) Near-Term Implementation Notes

1. Log every candidate from strategy layer (including rejected ones) with full feature snapshot.
2. Split metrics by regime and by rule variant.
3. Add a drift monitor job that evaluates rolling KPIs.
4. Add promotion policy: challenger must beat champion in recent out-of-sample window with risk constraints.
5. Integrate this loop into `train` + `forward` workflows incrementally.

## 14. How To Find the Optimal Data/Parameter Size (Conversation Notes)

If you're asking "how do I find the optimal amount of data (e.g., training window size / weighting)?" the only reliable answer in a drifting system (like markets) is:

Empirically, with walk-forward experiments and stability criteria.

Below is a clean engineering procedure to implement.

### 1) Define what "optimal" means first

Pick one primary objective, for example:

- Net PnL
- Sharpe / Sortino
- AUC / logloss (for the ML layer)
- Composite objective: `PnL - lambda * drawdown - mu * turnover`

Also track stability metrics:

- Performance variance across windows
- Worst-case drawdown
- Regime-to-regime consistency

Do not optimize only mean performance. Optimize robust forward performance.

### 2) Choose candidate data windows / weighting schemes

Test a grid of plausible options, for example:

Hard rolling windows:

- 1 month
- 3 months
- 6 months
- 12 months
- 24 months

Time-decay weighting:

- Half-life = 1 month
- Half-life = 3 months
- Half-life = 6 months
- Half-life = 12 months

You can test both and compare.

### 3) Use walk-forward evaluation (mandatory)

For each candidate window/weighting:

- Train on window `W`
- Test on next period `H` (for example next week/month)
- Slide forward:
  - Train: `[t0 - W, t0]`
  - Test: `(t0, t0 + H]`
- Repeat over many folds

Collect:

- Mean performance
- Standard deviation / downside risk
- Worst fold performance
- Stability across time

Never use random splits.

### 4) Score each option with a robust objective

Do not just pick the highest average return.

Use a robust score, for example:

`Score = Mean - alpha * Std - beta * MaxDrawdown`

Or use:

- Median fold performance
- 25th percentile fold performance
- Percent of folds profitable

This reduces overfitting to lucky windows.

### 5) Look for the bias-variance tradeoff curve

Common pattern:

- Very short window: adapts fast, but noisy/unstable/overfit
- Very long window: stable, but stale/slow to adapt
- Middle window: best forward robustness

Plot:

- Window size vs performance
- Window size vs variance/drawdown

Pick the knee of the curve, not the extreme.

### 6) Do this separately for each layer

- ML layer (`feature -> label` mapping)
- Rule layer parameters (thresholds, structure filters)
- Optionally per regime

It is normal for layers to prefer different windows.

### 7) Add a meta-rule: re-optimize the window periodically

Markets change, so re-run window selection every:

- 3 months or 6 months

Or trigger re-evaluation if live performance materially deviates from expected distribution.

Adaptation itself must be adaptive.

### 8) Fast starting heuristic (intraday futures)

Start with rolling windows:

- 1M, 3M, 6M, 12M

Working prior:

- ML layer often prefers 3-6 months
- Rule calibration often prefers 6-12 months

Then let walk-forward results decide.

### 9) Key principle

The "optimal" window is not universal.
It is the one that gives the best forward stability, not the highest backtest peak.

If you optimize for peak backtest, you will overfit the window itself.

### 10) Inputs needed to make this concrete

Provide:

- Instrument + timeframe
- Retraining frequency
- Data volume per month
- Main objective (PnL, Sharpe, AUC, etc.)

Then define:

- A specific window grid
- A walk-forward scheme
- A scoring function tailored to this system
