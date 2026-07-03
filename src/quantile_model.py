"""
Prediction intervals via quantile regression.

A point forecast answers "how much will we sell?" but a kitchen actually needs
"how much should we prep to hit a chosen service level?". We answer that by
training gradient-boosting models with the **pinball (quantile) loss** at several
quantiles. Prepping at the *q*-th quantile is expected to meet demand on a
fraction *q* of days — a principled, calibrated replacement for an ad-hoc
"+10% safety margin".

Outputs a bundle at ``models/quantile_model[_real].pkl`` holding one model per
quantile plus their measured test-set coverage.

Usage: ``python src/quantile_model.py --source {synthetic,real}``
"""

from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from datasets import get_config, resolve
from features import get_feature_columns, preprocess_data, train_test_split_by_date

# Lower/median/upper band + higher service levels used for prep decisions.
QUANTILES: list[float] = [0.1, 0.5, 0.8, 0.9, 0.95]

# Fixed, sensible hyperparameters. (The point model in train.py demonstrates the
# full RandomizedSearchCV tuning methodology; we keep these fixed for tractability
# since we fit one model per quantile.)
HGB_PARAMS = dict(learning_rate=0.05, max_iter=400, max_leaf_nodes=31, random_state=42)


def train_quantile_models(X, y, quantiles: list[float]) -> dict[float, object]:
    """Fit one HistGradientBoosting model per quantile using the pinball loss."""
    models = {}
    for q in quantiles:
        model = HistGradientBoostingRegressor(loss="quantile", quantile=q, **HGB_PARAMS)
        model.fit(X, y)
        models[q] = model
    return models


def predict_quantiles(df, bundle: dict) -> dict[float, np.ndarray]:
    """Predict every stored quantile for ``df``.

    Predictions are clipped at 0 and sorted across quantiles per row so the
    bands never cross (a known quirk of independently-fit quantile models).
    """
    features = bundle["features"]
    quantiles = bundle["quantiles"]
    preds = np.column_stack([bundle["models"][q].predict(df[features]) for q in quantiles])
    preds = np.clip(preds, 0, None)
    preds = np.sort(preds, axis=1)  # enforce q10 <= q50 <= q90 ...
    return {q: preds[:, i] for i, q in enumerate(quantiles)}


def pinball_loss(y_true, y_pred, q: float) -> float:
    """Average pinball (quantile) loss — the metric these models optimise."""
    diff = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def main(source: str = "synthetic") -> None:
    config = get_config(source)
    print(f"Training quantile models on '{source}' data...")
    df = preprocess_data(resolve(config["sales"]))
    train_df, test_df = train_test_split_by_date(
        df, split_date=config["split_date"], test_fraction=config["test_fraction"]
    )
    features = get_feature_columns(df)

    X_train, y_train = train_df[features], train_df["units_sold"]
    y_test = test_df["units_sold"]

    models = train_quantile_models(X_train, y_train, QUANTILES)
    bundle = {"models": models, "features": features, "quantiles": QUANTILES, "source": source}

    # Measure calibration: the share of test days on which actual demand fell at
    # or below the predicted quantile should be close to the quantile itself.
    q_preds = predict_quantiles(test_df, bundle)
    coverage, pinball = {}, {}
    print(f"{'quantile':>9} {'coverage':>9} {'pinball':>9}")
    for q in QUANTILES:
        coverage[q] = float(np.mean(y_test.to_numpy() <= q_preds[q]))
        pinball[q] = pinball_loss(y_test, q_preds[q], q)
        print(f"{q:>9.2f} {coverage[q]:>9.2%} {pinball[q]:>9.3f}")
    bundle["coverage"] = coverage
    bundle["pinball"] = pinball

    model_path = resolve(config["quantile_model"])
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(bundle, model_path, compress=3)
    print(f"Saved {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train quantile (prediction-interval) models.")
    parser.add_argument("--source", choices=["synthetic", "real"], default="synthetic")
    args = parser.parse_args()
    main(source=args.source)
