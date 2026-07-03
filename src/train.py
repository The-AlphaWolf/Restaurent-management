"""
Model training and evaluation.

Trains and tunes two tree-based regressors to forecast per-item daily demand:
    1. Random Forest
    2. Histogram Gradient Boosting (scikit-learn's fast gradient-boosting)

Both are pure scikit-learn, so the project has **no native (C/OpenMP)
dependencies** and runs identically on Windows, Linux, and Streamlit Cloud.

Usage: ``python src/train.py --source {synthetic,real}``

Key methodology (the things an interviewer will ask about):
- **Time-based split**: train on the earlier dates, test on the most recent
  ones (per the dataset registry). We never shuffle time series data.
- **Time-series cross-validation** (``TimeSeriesSplit``) inside
  ``RandomizedSearchCV`` so hyperparameters are chosen without peeking ahead.
- **Naive baseline** (predict last week's value) to prove the model adds value.
- Metrics reported: MAE, RMSE, MAPE, R2.
"""

from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from datasets import get_config, resolve
from features import get_feature_columns, preprocess_data, train_test_split_by_date

# --- Hyperparameter search spaces (documented so they are reproducible) ---
RF_PARAM_DIST: dict[str, list] = {
    "n_estimators": [100, 200, 300],
    "max_depth": [8, 12, 16, None],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", 0.5, 1.0],
}

HGB_PARAM_DIST: dict[str, list] = {
    "learning_rate": [0.03, 0.05, 0.1],
    "max_iter": [200, 400, 600],
    "max_depth": [None, 6, 10],
    "max_leaf_nodes": [31, 63],
    "l2_regularization": [0.0, 0.1, 1.0],
}


def evaluate_model(name: str, y_true, y_pred) -> dict[str, float]:
    """Compute and print MAE, RMSE, MAPE, R2 for a set of predictions."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # sklearn returns a fraction; multiply by 100 for a percentage.
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)

    print(f"--- {name} ---")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R2:   {r2:.4f}")
    print("-" * 24)

    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


def tune_model(estimator, param_dist: dict, X, y, n_iter: int = 15):
    """Randomized hyperparameter search with time-series cross-validation.

    ``TimeSeriesSplit`` ensures every validation fold is *after* its training
    fold, so we never validate on the past using knowledge of the future.
    """
    tscv = TimeSeriesSplit(n_splits=3)
    search = RandomizedSearchCV(
        estimator,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=tscv,
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def main(source: str = "synthetic") -> None:
    config = get_config(source)
    print(f"Loading and preprocessing '{source}' data ({config['label']})...")
    df = preprocess_data(resolve(config["sales"]))

    train_df, test_df = train_test_split_by_date(
        df, split_date=config["split_date"], test_fraction=config["test_fraction"]
    )
    features = get_feature_columns(df)

    X_train, y_train = train_df[features], train_df["units_sold"]
    X_test, y_test = test_df[features], test_df["units_sold"]

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape:     {X_test.shape}")
    print(f"Features ({len(features)}): {features}\n")

    # --- Naive baseline: "same as last week" (lag_7). ---
    print("Evaluating Baseline (same as last week)...")
    baseline_metrics = evaluate_model("Baseline (lag_7)", y_test, test_df["lag_7"])

    # --- Random Forest (tuned). ---
    print("\nTuning Random Forest...")
    rf_model, rf_params = tune_model(
        RandomForestRegressor(random_state=42, n_jobs=-1), RF_PARAM_DIST, X_train, y_train
    )
    print(f"Best RF params: {rf_params}")
    rf_metrics = evaluate_model("Random Forest", y_test, rf_model.predict(X_test))

    # --- Histogram Gradient Boosting (tuned). ---
    print("\nTuning Gradient Boosting...")
    hgb_model, hgb_params = tune_model(
        HistGradientBoostingRegressor(random_state=42), HGB_PARAM_DIST, X_train, y_train
    )
    print(f"Best GB params: {hgb_params}")
    hgb_metrics = evaluate_model("Gradient Boosting", y_test, hgb_model.predict(X_test))

    # --- Select the better model by test MAE. ---
    if hgb_metrics["mae"] <= rf_metrics["mae"]:
        best_model, best_name, best_params = hgb_model, "Gradient Boosting", hgb_params
    else:
        best_model, best_name, best_params = rf_model, "Random Forest", rf_params

    improvement = (baseline_metrics["mae"] - min(rf_metrics["mae"], hgb_metrics["mae"]))
    pct = improvement / baseline_metrics["mae"] * 100
    print(f"\nBest model: {best_name}")
    print(f"MAE improvement vs naive baseline: {improvement:.2f} units ({pct:.1f}%)")

    # --- Permutation importance on the test set. ---
    # Works for any estimator (unlike ``feature_importances_``, which
    # HistGradientBoosting lacks) and measures real predictive contribution.
    print("\nComputing permutation importance...")
    perm = permutation_importance(
        best_model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
    )
    feature_importance = sorted(
        zip(features, perm.importances_mean, strict=True), key=lambda kv: kv[1], reverse=True
    )

    # --- Persist the winning model + everything needed to reuse it. ---
    model_path = resolve(config["model"])
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(
        {
            "model": best_model,
            "features": features,
            "model_name": best_name,
            "best_params": best_params,
            "feature_importance": feature_importance,
            "source": source,
            "metrics": {
                "baseline": baseline_metrics,
                "random_forest": rf_metrics,
                "gradient_boosting": hgb_metrics,
            },
        },
        model_path,
        compress=3,  # tree ensembles compress ~4x -> keeps the repo lean
    )
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the demand-prediction model.")
    parser.add_argument(
        "--source",
        choices=["synthetic", "real"],
        default="synthetic",
        help="Which dataset to train on (default: synthetic).",
    )
    args = parser.parse_args()
    main(source=args.source)
