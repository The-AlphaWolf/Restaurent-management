"""
Prediction helper: score a preprocessed frame with a loaded model bundle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def predict_demand(df: pd.DataFrame, model_info: dict) -> np.ndarray:
    """Predict non-negative demand for a preprocessed dataframe.

    Args:
        df: Feature dataframe containing every column the model was trained on.
        model_info: Model bundle from ``joblib.load`` (model + feature list).

    Returns:
        Array of predicted demand, clipped at 0 (demand cannot be negative).
    """
    model = model_info["model"]
    features = model_info["features"]

    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features in dataframe: {missing_features}")

    predictions = model.predict(df[features])
    return np.clip(predictions, 0, None)
