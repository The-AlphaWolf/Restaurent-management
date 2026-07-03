"""
Feature engineering for the demand-prediction model.

Turns the raw daily sales table into a model-ready feature matrix. Every
feature is derived only from information that would be available *before* the
day being predicted, so there is no target leakage from the future.

Feature groups and why each is included:
- Lag features (lag_1/7/14):      demand is autocorrelated; yesterday, last
                                  week, and two weeks ago are strong signals.
- Rolling means (7/14, shifted):  smooth the recent demand level and dampen
                                  day-to-day noise. Shifted by 1 so the target
                                  day is never part of its own average.
- Calendar (day_of_week, month):  capture weekly and yearly seasonality that
                                  the synthetic data was built around.
- is_weekend / is_holiday:        explicit flags for the biggest demand spikes.
- trend_index:                    a monotonic day counter that lets tree models
                                  learn the gradual growth trend over time.
- temperature / price:            environmental / item drivers already in the raw data.
- category (one-hot) + item id:   identify *which* item/category a row belongs to.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

# Columns that must never be fed to the model as features (identifiers / target).
NON_FEATURE_COLUMNS: list[str] = ["date", "item_id", "units_sold"]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag, rolling, and calendar features to the raw sales frame.

    Args:
        df: Raw sales data with at least ``date``, ``item_id`` and
            ``units_sold`` columns.

    Returns:
        A new dataframe with engineered features and the leading rows (which
        have undefined lags) dropped.
    """
    # Sort chronologically *within* each item so lags/rolls align per item.
    df = df.sort_values(by=["item_id", "date"]).reset_index(drop=True)

    grouped = df.groupby("item_id")["units_sold"]

    # --- Lag features: demand N days ago (per item). ---
    df["lag_1"] = grouped.shift(1)
    df["lag_7"] = grouped.shift(7)
    df["lag_14"] = grouped.shift(14)

    # --- Rolling means: average of the *previous* N days (shift(1) excludes today). ---
    df["rolling_mean_7"] = grouped.transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()
    )
    df["rolling_mean_14"] = grouped.transform(
        lambda x: x.shift(1).rolling(window=14, min_periods=1).mean()
    )

    # --- Calendar features derived from the date (single source of truth,
    #     so they work identically for the synthetic and real datasets). ---
    df["day_of_week"] = df["date"].dt.dayofweek          # 0=Mon .. 6=Sun (weekly seasonality)
    df["month"] = df["date"].dt.month                    # 1..12 (yearly seasonality)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)  # Sat/Sun flag
    # Monotonic day counter from the first date in the dataset -> lets trees
    # split on "how far into the restaurant's life" a row is (growth trend).
    df["trend_index"] = (df["date"] - df["date"].min()).dt.days

    # Drop the first rows per item where the longest lag (14) is undefined.
    df = df.dropna().reset_index(drop=True)

    return df


def preprocess_data(filepath: str = "data/sales_data.csv") -> pd.DataFrame:
    """Load raw sales data, engineer features, and encode categoricals.

    Args:
        filepath: Path to ``sales_data.csv``.

    Returns:
        A model-ready dataframe. Feature columns are everything except the
        identifiers/target listed in :data:`NON_FEATURE_COLUMNS`.
    """
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])

    df = create_features(df)

    # ``is_holiday`` needs an external calendar; default to 0 if a dataset
    # does not provide it, so the feature column always exists.
    if "is_holiday" not in df.columns:
        df["is_holiday"] = 0

    # One-hot encode category (if present) for an interpretable
    # "which category" signal.
    if "category" in df.columns:
        df = pd.get_dummies(df, columns=["category"], drop_first=True)

    # Integer code per item so a single global model can distinguish items
    # while still sharing learned seasonality/lag patterns across the menu.
    df["item_id_encoded"] = df["item_id"].astype("category").cat.codes

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the model feature columns (all columns bar identifiers/target)."""
    return [col for col in df.columns if col not in NON_FEATURE_COLUMNS]


def get_test_cutoff(
    df: pd.DataFrame, split_date: str | None = None, test_fraction: float | None = 0.2
) -> pd.Timestamp:
    """Return the chronological date that separates train from test.

    Either honour an explicit ``split_date`` or hold out the most recent
    ``test_fraction`` of unique dates. Splitting on dates (never rows) keeps a
    single day entirely within one side of the split.
    """
    if split_date is not None:
        return pd.Timestamp(split_date)
    unique_dates = np.sort(df["date"].unique())
    return pd.Timestamp(unique_dates[int(len(unique_dates) * (1 - test_fraction))])


def train_test_split_by_date(
    df: pd.DataFrame, split_date: str | None = None, test_fraction: float | None = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df`` chronologically into (train, test)."""
    cutoff = get_test_cutoff(df, split_date, test_fraction)
    return df[df["date"] < cutoff].copy(), df[df["date"] >= cutoff].copy()


if __name__ == "__main__":
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "sales_data.csv"
    )
    if os.path.exists(data_path):
        processed_df = preprocess_data(data_path)
        print(f"Processed dataframe shape: {processed_df.shape}")
        print("Feature columns:", get_feature_columns(processed_df))
        print(processed_df.head())
    else:
        print(f"File not found: {data_path}. Run data_generation.py first.")
