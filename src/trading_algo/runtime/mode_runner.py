from __future__ import annotations

import os
from dataclasses import dataclass

from trading_algo.backtest import BacktestConfig, load_bars_from_csv, run_backtest as run_backtest_sim
from trading_algo.config import RuntimeConfig, env_bool, env_int, load_runtime_config, must_env
from trading_algo.ml import train_xgboost_from_csv
from trading_algo.runtime.bot_runtime import run as run_forward_runtime
from trading_algo.strategy import OneShotLongStrategy


@dataclass(frozen=True)
class ModeOptions:
    mode: str
    data_csv: str | None
    strategy: str
    model_out: str
    hold_bars: int


def _strategy_from_name(name: str, hold_bars: int):
    key = name.strip().lower()
    if key in {"oneshot", "one_shot", "one-shot"}:
        return OneShotLongStrategy(hold_bars=hold_bars)
    raise ValueError(f"Unsupported strategy: {name}")


def _resolve_data_csv(explicit: str | None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    return must_env("BACKTEST_DATA_CSV")


def run_forward(config: RuntimeConfig) -> None:
    enabled = env_bool("BOT_ENABLED", False)
    environment = (os.getenv("TRADING_ENVIRONMENT") or "DEMO").strip()
    print(f"TRADING_ENVIRONMENT = {environment}")
    print(f"BOT_ENABLED        = {enabled}")
    print(f"SYMBOL             = {config.symbol}")
    print(f"ACCOUNT_ID         = {config.account_id}")
    print(f"LIVE               = {config.live}")
    print(f"TRADE_ON_START     = {config.trade_on_start}")
    if not enabled:
        print("BOT_ENABLED=0 -> Trading disabled.")
        return
    run_forward_runtime(config)


def run_backtest(data_csv: str, strategy_name: str, hold_bars: int) -> None:
    bars = load_bars_from_csv(data_csv)
    strategy = _strategy_from_name(strategy_name, hold_bars)
    cfg = BacktestConfig(
        initial_cash=float(env_int("BACKTEST_INITIAL_CASH", 10_000)),
        fee_per_order=float(env_int("BACKTEST_FEE_PER_ORDER", 1)),
        slippage_bps=float(env_int("BACKTEST_SLIPPAGE_BPS", 1)),
    )
    result = run_backtest_sim(bars, strategy, cfg)
    print("BACKTEST RESULT")
    print(f"bars={len(bars)} trades={result.num_trades}")
    print(f"final_equity={result.final_equity:.2f}")
    print(f"net_pnl={result.net_pnl:.2f} return_pct={result.total_return_pct:.2f}")
    print(f"win_rate_pct={result.win_rate_pct:.2f} max_drawdown_pct={result.max_drawdown_pct:.2f}")
def run_train(data_csv: str, model_out: str) -> None:
    train_xgboost_from_csv(data_csv, model_out)


def run_mode(options: ModeOptions) -> None:
    mode = options.mode.strip().lower()
    if mode == "forward":
        run_forward(load_runtime_config())
        return

    data_csv = _resolve_data_csv(options.data_csv)
    if mode == "backtest":
        run_backtest(data_csv, options.strategy, options.hold_bars)
        return
    if mode == "train":
        run_train(data_csv, options.model_out)
        return
    raise ValueError(f"Unsupported mode: {options.mode}. Use forward, backtest, or train.")
