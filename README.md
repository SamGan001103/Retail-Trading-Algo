# Retail-Trading-Algo

# Retail Trading Algo (ProjectX / TopstepX)

This repository is a **Python-based execution and realtime infrastructure** for building an automated trading system on the ProjectX / TopstepX API.

At the current stage, this project is **not a strategy**. It is a **reliable execution + state foundation** that:
- Authenticates with ProjectX
- Streams realtime orders / positions / market data via SignalR
- Places market orders with OCO brackets (SL/TP)
- Monitors state and enforces safety rules
- Provides kill-switch / flatten functionality

Think of this as the **engine room** of the trading bot. Strategy logic will be layered on top.

---

## ✅ Current Capabilities

### 1. REST API Integration
- Login and JWT token handling
- Authenticated POST helpers
- Endpoints implemented:
  - Search open orders
  - Search open positions
  - Place orders (market + brackets)
  - Cancel orders
  - Close positions
  - Flatten account

Location:
- `core/projectx_api.py`
- `core/execution_engine.py`

---

### 2. Execution Engine
The execution engine provides:
- Token caching (TTL-based) to avoid repeated logins
- Safety gate: prevents entering if already in a position or if stale orders exist
- Order placement:
  - Market entry
  - Automatic SL/TP bracket orders (signed tick offsets for long/short)
- Emergency controls:
  - Cancel individual orders
  - Close contracts
  - Flatten all (cancel orders + close positions)
- Verification:
  - Confirms bracket orders exist after entry

Location:
- `core/execution_engine.py`

---

### 3. Realtime Streaming (SignalR)
- Connects to ProjectX SignalR hubs
- Subscribes to:
  - Orders
  - Positions
  - Quotes
  - Trades (optional depth)
- Maintains in-memory state:
  - `orders`
  - `positions`
  - `last_quote_by_contract`
  - `last_trade_by_contract`
- Includes a custom **Record Separator (0x1E) framing fix** to prevent partial JSON decode errors (important for stability)

Location:
- `core/realtime_client.py`

---

### 4. Bot Runtime Harness
A simple runtime loop that:
- Resolves `contract_id` from symbol
- Starts realtime streaming
- Optionally flattens on startup
- Monitors:
  - Whether you are flat or in a position
  - Whether brackets exist
- If configured:
  - Places exactly **one** market trade with brackets (`TRADE_ON_START=true`)
- If brackets are missing:
  - Triggers emergency `flatten()`

This is currently a **safety + plumbing test harness**, not a strategy.

Location:
- `bot/bot_runtime.py`

---

### 5. Utility Scripts
- Account lookup & validation:
  - `account/account_lookup.py`
  - `account/account_check.py`
- Market / contract lookup:
  - `market/market_lookup.py`
- Order tools:
  - `orders/order_place.py`
  - `orders/order_cancel.py`
  - `orders/orders_open.py`
- Position tools:
  - `positions/positions_open.py`
  - `positions/position_close_contract.py`
  - `positions/flatten_all.py`

These are useful for **manual testing and debugging**.

---

## ⚠️ Important Security Note

**DO NOT commit your real API keys.**

- Your `.env` must stay local and ignored by git
- Use `.env.example` with placeholders only
- If a key was ever committed, **rotate/revoke it immediately**

---

## 🧱 What This Repo Is (and Isn’t)

### It IS:
- A working execution layer
- A realtime state ingestion layer
- A safety framework (brackets, flatten, entry gating)
- A foundation for a serious trading system

### It is NOT (yet):
- A trading strategy
- A signal generation system
- A backtesting framework
- A portfolio/risk engine

---

## 🗺️ Roadmap (In Order, Recommended)

### Step 1 — Project Hygiene (Do this first)
- Convert to a proper package layout (remove `sys.path.insert` hacks)
- Unify REST helpers (remove duplicated login/post code in scripts)
- Fix env var naming inconsistencies (`ACCOUNT_ID` vs `PROJECTX_ACCOUNT_ID`)
- Update `requirements.txt` to include:
  - `requests`
  - `signalrcore`
  - `python-dotenv`

**Why:** Stability, maintainability, zero latency impact.

---

### Step 2 — Make Realtime the Primary State Source
- Use `RealtimeClient` state as the main decision input
- Use REST only for:
  - Startup sync
  - Reconnect recovery
  - Inconsistency checks

**Why:** REST is slow. SignalR is your low-latency data path.

---

### Step 3 — Add a Strategy Interface
Introduce something like:
- `Strategy.on_tick(market_event, state) -> signal`
- `RiskManager.size(signal, state) -> position_size`
- `ExecutionPolicy.execute(signal)`

Start simple:
- Time-based entry
- Breakout
- Mean reversion
- One contract only

**Why:** This is where the bot becomes an actual trading system.

---

### Step 4 — Add Logging & Audit Trail
- Log:
  - Orders sent
  - Fills
  - Position changes
  - State transitions
- Use structured logs (JSONL or similar)
- Keep logging **off the hot path** (buffered writes)

**Why:** You can’t debug or trust a trading system without an audit trail.

---

### Step 5 — Add Hard Risk Limits
- Max daily loss
- Max trades per day
- Cooldown after loss
- Stale data detection (no ticks → no trading)
- Global kill switch

**Why:** This is what keeps you alive when something breaks.

---

## 🎯 Design Philosophy

- **Realtime first, REST second**
- **Safety before strategy**
- **Execution must be boring and reliable**
- **No blocking calls in the decision loop**
- **Latency is only spent where it earns money**

---

## 🚀 Current Status

- Execution: ✅ Working  
- Realtime streaming: ✅ Working  
- Safety (brackets, flatten, gating): ✅ Working  
- Strategy: ❌ Not implemented yet  
- Backtesting: ❌ Not implemented  
- Risk framework: ⚠️ Minimal (only structural checks)
