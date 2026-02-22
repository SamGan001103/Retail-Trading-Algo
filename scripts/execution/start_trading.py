import argparse

from trading_algo.runtime import ModeOptions, run_mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Master launcher for forward, backtest, and train modes.")
    parser.add_argument("--mode", choices=["forward", "backtest", "train"], default="forward")
    parser.add_argument("--profile", choices=["normal", "debug"], default=None, help="Runtime profile.")
    parser.add_argument(
        "--data-parquet",
        default=None,
        help="Historical parquet dataset path for backtest/train modes.",
    )
    parser.add_argument("--strategy", default="oneshot", help="Strategy key (e.g., oneshot, ny_structure).")
    parser.add_argument("--model-out", default="artifacts/models/xgboost_model.json")
    parser.add_argument("--hold-bars", type=int, default=20, help="Hold duration for oneshot strategy.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_mode(
        ModeOptions(
            mode=args.mode,
            profile=args.profile,
            data_path=args.data_parquet,
            strategy=args.strategy,
            model_out=args.model_out,
            hold_bars=args.hold_bars,
        )
    )


if __name__ == "__main__":
    main()
