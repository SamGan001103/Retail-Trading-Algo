from __future__ import annotations

from pathlib import Path

from trading_algo.backtest import load_bars_from_csv


def _build_features_and_labels(path: str) -> tuple[list[list[float]], list[int]]:
    bars = load_bars_from_csv(path)
    if len(bars) < 5:
        raise RuntimeError("Need at least 5 bars to build training features")

    x: list[list[float]] = []
    y: list[int] = []
    for i in range(3, len(bars) - 1):
        b0 = bars[i]
        b1 = bars[i - 1]
        b2 = bars[i - 2]
        b3 = bars[i - 3]
        next_bar = bars[i + 1]

        ret_1 = (b0.close - b1.close) / b1.close if b1.close else 0.0
        ret_2 = (b1.close - b2.close) / b2.close if b2.close else 0.0
        ret_3 = (b2.close - b3.close) / b3.close if b3.close else 0.0
        range_pct = (b0.high - b0.low) / b0.close if b0.close else 0.0
        vol_chg = (b0.volume - b1.volume) / b1.volume if b1.volume else 0.0
        x.append([ret_1, ret_2, ret_3, range_pct, vol_chg])
        y.append(1 if next_bar.close > b0.close else 0)
    return x, y


def train_xgboost_from_csv(data_csv: str, model_out: str) -> None:
    x, y = _build_features_and_labels(data_csv)
    split = max(1, int(len(x) * 0.8))
    x_train, x_valid = x[:split], x[split:]
    y_train, y_valid = y[:split], y[split:]

    try:
        import xgboost as xgb  # type: ignore
    except Exception:
        print("xgboost is not installed. Install it with: pip install xgboost")
        print(f"Prepared dataset rows: train={len(x_train)} valid={len(x_valid)}")
        return

    dtrain = xgb.DMatrix(x_train, label=y_train)
    dvalid = xgb.DMatrix(x_valid, label=y_valid) if x_valid else None

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.05,
        "max_depth": 4,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "seed": 42,
    }
    watchlist = [(dtrain, "train")]
    if dvalid is not None:
        watchlist.append((dvalid, "valid"))
    booster = xgb.train(params, dtrain, num_boost_round=200, evals=watchlist, verbose_eval=False)

    out_path = Path(model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out_path))
    print(f"Model saved: {out_path}")
    print(f"Rows used: train={len(x_train)} valid={len(x_valid)}")

