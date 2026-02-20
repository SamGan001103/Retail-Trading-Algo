from __future__ import annotations

import os
from dataclasses import dataclass

from trading_algo.backtest import BacktestConfig, load_bars_from_csv, run_backtest as run_backtest_sim
from trading_algo.config import RuntimeConfig, env_bool, env_float, env_int, get_symbol_profile, load_runtime_config, must_env
from trading_algo.ml import SetupMLGate, train_xgboost_from_csv
from trading_algo.runtime.bot_runtime import run as run_forward_runtime
from trading_algo.strategy import NYSessionMarketStructureStrategy, OneShotLongStrategy


@dataclass(frozen=True)
class ModeOptions:
    mode: str
    data_csv: str | None
    strategy: str
    model_out: str
    hold_bars: int


def _strategy_from_name(name: str, hold_bars: int, *, for_forward: bool):
    key = name.strip().lower()
    if key in {"oneshot", "one_shot", "one-shot"}:
        return OneShotLongStrategy(hold_bars=hold_bars)
    if key in {"ny_structure", "ny_session", "market_structure", "mnq_ny"}:
        entry_mode_default = "tick" if for_forward else "bar"
        symbol = (os.getenv("SYMBOL") or "MNQ").strip().upper()
        profile = get_symbol_profile(symbol)
        sizing_drawdown = env_float("STRAT_ACCOUNT_MAX_DRAWDOWN", env_float("ACCOUNT_MAX_DRAWDOWN", 2_500.0))
        strategy = NYSessionMarketStructureStrategy(
            size=env_int("SIZE", 1),
            session_start=(os.getenv("STRAT_NY_SESSION_START") or "09:30").strip(),
            session_end=(os.getenv("STRAT_NY_SESSION_END") or "16:00").strip(),
            tz_name=(os.getenv("STRAT_TZ_NAME") or "America/New_York").strip(),
            htf_aggregation=env_int("STRAT_HTF_AGGREGATION", 5),
            htf_swing_strength_high=env_int("STRAT_HTF_SWING_HIGH", 5),
            htf_swing_strength_low=env_int("STRAT_HTF_SWING_LOW", 5),
            ltf_swing_strength_high=env_int("STRAT_LTF_SWING_HIGH", 3),
            ltf_swing_strength_low=env_int("STRAT_LTF_SWING_LOW", 3),
            wick_sweep_ratio_min=env_float("STRAT_SWEEP_WICK_MIN", 0.5),
            sweep_expiry_bars=env_int("STRAT_SWEEP_EXPIRY_BARS", 40),
            equal_level_tolerance_bps=env_float("STRAT_EQUAL_LEVEL_TOL_BPS", 8.0),
            key_area_tolerance_bps=env_float("STRAT_KEY_AREA_TOL_BPS", 12.0),
            min_confluence_score=env_int("STRAT_MIN_CONFLUENCE", 1),
            require_orderflow=env_bool("STRAT_REQUIRE_ORDERFLOW", False),
            entry_mode=(os.getenv("STRAT_ENTRY_MODE") or entry_mode_default).strip().lower(),
            max_hold_bars=hold_bars,
            tick_setup_expiry_bars=env_int("STRAT_TICK_SETUP_EXPIRY_BARS", 3),
            tick_history_size=env_int("STRAT_TICK_HISTORY_SIZE", 120),
            tick_min_imbalance=env_float("STRAT_TICK_MIN_IMBALANCE", 0.12),
            tick_min_trade_size=env_float("STRAT_TICK_MIN_TRADE_SIZE", 1.0),
            tick_spoof_collapse_ratio=env_float("STRAT_TICK_SPOOF_COLLAPSE", 0.35),
            tick_absorption_min_trades=env_int("STRAT_TICK_ABSORPTION_TRADES", 2),
            tick_iceberg_min_reloads=env_int("STRAT_TICK_ICEBERG_RELOADS", 2),
            symbol=symbol,
            tick_size=env_float("STRAT_TICK_SIZE", profile.tick_size),
            tick_value=env_float("STRAT_TICK_VALUE", profile.tick_value),
            account_max_drawdown=sizing_drawdown,
            max_trade_drawdown_fraction=env_float("STRAT_MAX_TRADE_DRAWDOWN_FRACTION", 0.15),
            risk_min_rrr=env_float("STRAT_MIN_RRR", 3.0),
            risk_max_rrr=env_float("STRAT_MAX_RRR", 10.0),
            sl_noise_buffer_ticks=env_int("STRAT_SL_NOISE_BUFFER_TICKS", 2),
            sl_max_ticks=env_int("STRAT_SL_MAX_TICKS", 200),
            tp_front_run_ticks=env_int("STRAT_TP_FRONT_RUN_TICKS", 2),
            dom_liquidity_wall_size=env_float("STRAT_DOM_LIQUIDITY_WALL_SIZE", profile.dom_liquidity_wall_size),
            ml_min_size_fraction=env_float("STRAT_ML_MIN_SIZE_FRACTION", 0.35),
            ml_size_floor_score=env_float("STRAT_ML_SIZE_FLOOR_SCORE", 0.55),
            ml_size_ceiling_score=env_float("STRAT_ML_SIZE_CEILING_SCORE", 0.90),
            enable_exhaustion_market_exit=env_bool("STRAT_ENABLE_EXHAUSTION_MARKET_EXIT", True),
        )
        if env_bool("STRAT_ML_GATE_ENABLED", False):
            strategy.set_ml_gate(
                SetupMLGate(
                    enabled=True,
                    model_path=(os.getenv("STRAT_ML_MODEL_PATH") or "").strip(),
                    min_proba=env_float("STRAT_ML_MIN_PROBA", 0.55),
                    fail_open=env_bool("STRAT_ML_FAIL_OPEN", False),
                )
            )
        return strategy
    raise ValueError(f"Unsupported strategy: {name}")


def _resolve_data_csv(explicit: str | None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    return must_env("BACKTEST_DATA_CSV")


def run_forward(config: RuntimeConfig, strategy_name: str, hold_bars: int) -> None:
    enabled = env_bool("BOT_ENABLED", False)
    environment = (os.getenv("TRADING_ENVIRONMENT") or "DEMO").strip()
    strategy = _strategy_from_name(strategy_name, hold_bars, for_forward=True)
    print(f"TRADING_ENVIRONMENT = {environment}")
    print(f"BROKER             = {config.broker}")
    print(f"BOT_ENABLED        = {enabled}")
    print(f"SYMBOL             = {config.symbol}")
    print(f"ACCOUNT_ID         = {config.account_id}")
    print(f"LIVE               = {config.live}")
    print(f"STRATEGY           = {strategy_name}")
    print(f"TRADE_ON_START     = {config.trade_on_start}")
    if not enabled:
        print("BOT_ENABLED=0 -> Trading disabled.")
        return
    run_forward_runtime(config, strategy=strategy)


def run_backtest(data_csv: str, strategy_name: str, hold_bars: int) -> None:
    bars = load_bars_from_csv(data_csv)
    strategy = _strategy_from_name(strategy_name, hold_bars, for_forward=False)
    symbol = (os.getenv("SYMBOL") or "MNQ").strip().upper()
    profile = get_symbol_profile(symbol)
    max_drawdown_abs = env_float("ACCOUNT_MAX_DRAWDOWN_KILLSWITCH", env_float("ACCOUNT_MAX_DRAWDOWN", 0.0))
    cfg = BacktestConfig(
        initial_cash=env_float("BACKTEST_INITIAL_CASH", 10_000.0),
        fee_per_order=env_float("BACKTEST_FEE_PER_ORDER", 1.0),
        slippage_bps=env_float("BACKTEST_SLIPPAGE_BPS", 1.0),
        tick_size=env_float("STRAT_TICK_SIZE", profile.tick_size),
        max_drawdown_abs=max_drawdown_abs if max_drawdown_abs > 0 else None,
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
        run_forward(load_runtime_config(), options.strategy, options.hold_bars)
        return

    data_csv = _resolve_data_csv(options.data_csv)
    if mode == "backtest":
        run_backtest(data_csv, options.strategy, options.hold_bars)
        return
    if mode == "train":
        run_train(data_csv, options.model_out)
        return
    raise ValueError(f"Unsupported mode: {options.mode}. Use forward, backtest, or train.")
