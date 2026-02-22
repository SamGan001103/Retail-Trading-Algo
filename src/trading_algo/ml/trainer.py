from __future__ import annotations

from pathlib import Path
from typing import Any


_FEATURE_COLUMNS = [
    "has_recent_sweep",
    "bias_ok",
    "continuation",
    "reversal",
    "equal_levels",
    "fib_retracement",
    "key_area_proximity",
    "confluence_score",
    "of_imbalance",
    "of_top_bid_size",
    "of_top_ask_size",
    "of_best_bid",
    "of_best_ask",
    "of_spread",
    "of_trade_size",
    "of_trade_price",
    "ml_shadow_score",
]


def _coerce_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _coerce_label(row: dict[str, list[Any]], idx: int) -> int | None:
    win_values = row.get("win")
    if win_values is not None:
        raw = win_values[idx]
        if raw is None:
            return None
        if isinstance(raw, bool):
            return 1 if raw else 0
        try:
            return 1 if int(raw) > 0 else 0
        except (TypeError, ValueError):
            return None
    result_label = row.get("result_label")
    if result_label is None:
        return None
    text = str(result_label[idx] or "").strip().lower()
    if text == "win":
        return 1
    if text in {"loss", "flat"}:
        return 0
    return None


def _build_features_and_labels(path: str) -> tuple[list[list[float]], list[int]]:
    try:
        import pyarrow.dataset as ds  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - runtime dependency path
        raise RuntimeError("Training from parquet requires `pyarrow`. Install with `py -3.11 -m pip install pyarrow`.") from exc

    dataset = ds.dataset(path, format="parquet")
    table = dataset.to_table()
    if table.num_rows <= 0:
        raise RuntimeError(f"No rows found in parquet dataset: {path}")
    data = table.to_pydict()
    rows = int(table.num_rows)

    x: list[list[float]] = []
    y: list[int] = []
    for idx in range(rows):
        label = _coerce_label(data, idx)
        if label is None:
            continue
        features = []
        for col in _FEATURE_COLUMNS:
            values = data.get(col)
            value = values[idx] if values is not None else None
            features.append(_coerce_float(value))
        x.append(features)
        y.append(label)

    if len(x) < 20:
        raise RuntimeError(
            f"Need at least 20 labeled rows for training, found {len(x)} in parquet dataset: {path}"
        )
    return x, y


def train_xgboost_from_parquet(data_path: str, model_out: str) -> None:
    x, y = _build_features_and_labels(data_path)
    split = max(1, int(len(x) * 0.8))
    x_train, x_valid = x[:split], x[split:]
    y_train, y_valid = y[:split], y[split:]

    try:
        import xgboost as xgb  # type: ignore[import-not-found]
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
