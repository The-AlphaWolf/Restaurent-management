"""
Prediction helpers: load the saved model artifact and score new data.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd


def load_model(model_path: str | None = None) -> dict:
    """Load the pickled model bundle (model + feature list + metadata)."""
    if model_path is None:
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "models", "best_model.pkl"
        )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Please run train.py first."
        )

    return joblib.load(model_path)


def predict_demand(df: pd.DataFrame, model_info: dict | None = None) -> np.ndarray:
    """Predict non-negative demand for a preprocessed dataframe.

    Args:
        df: Feature dataframe containing every column the model was trained on.
        model_info: Loaded model bundle; loaded from disk if not provided.

    Returns:
        Array of predicted demand, clipped at 0 (demand cannot be negative).
    """
    if model_info is None:
        model_info = load_model()

    model = model_info["model"]
    features = model_info["features"]

    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features in dataframe: {missing_features}")

    predictions = model.predict(df[features])
    return np.clip(predictions, 0, None)
