import argparse

from trading_algo.runtime import ModeOptions, run_mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Master launcher for forward, backtest, and train modes.")
    parser.add_argument("--mode", choices=["forward", "backtest", "train"], default="forward")
    parser.add_argument("--data-csv", default=None, help="Historical data CSV path for backtest/train modes.")
    parser.add_argument("--strategy", default="oneshot", help="Strategy key (e.g., oneshot, ny_structure).")
    parser.add_argument("--model-out", default="artifacts/models/xgboost_model.json")
    parser.add_argument("--hold-bars", type=int, default=20, help="Hold duration for oneshot strategy.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_mode(
        ModeOptions(
            mode=args.mode,
            data_csv=args.data_csv,
            strategy=args.strategy,
            model_out=args.model_out,
            hold_bars=args.hold_bars,
        )
    )


if __name__ == "__main__":
    main()
