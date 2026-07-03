"""
Multi-step (7-day-ahead) demand forecasting.

The one-day model uses *yesterday's actual* demand — unavailable when a manager
plans a whole week. This module forecasts each of the next 7 days using the
**direct** strategy: a single model predicts demand at day ``t + h`` from
information known at the origin day ``t``, with the horizon ``h`` as a feature.

No leakage: recent-demand features are computed as of the origin ``t`` (known),
and the target day's calendar (day-of-week, month, holiday) is deterministic and
therefore also known in advance. Only the label comes from the future.

Baseline: the **seasonal-naive** forecast (same weekday last week), which for a
≤7-day horizon is known at the origin and is a genuinely hard baseline to beat.

Usage: ``python src/multistep.py --source {synthetic,real}``
"""

from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from datasets import get_config, resolve
from features import get_test_cutoff

HORIZONS = list(range(1, 8))  # forecast 1..7 days ahead

# Recent-demand features available at the origin day t (all known when planning).
BASE_FEATURES = [
    "o_demand",       # demand on the origin day
    "o_lag6",         # demand a week before the origin
    "o_roll7",        # mean demand over the last 7 days
    "o_roll14",       # mean demand over the last 14 days
    "horizon",        # how many days ahead we are predicting
    "t_dow",          # target day: day of week
    "t_month",        # target day: month
    "t_is_weekend",   # target day: weekend flag
    "t_is_holiday",   # target day: holiday flag
]

HGB_PARAMS = dict(learning_rate=0.05, max_iter=400, max_leaf_nodes=31, random_state=42)


def _origin_features(sales_df: pd.DataFrame) -> pd.DataFrame:
    """Attach origin-day recent-demand features (per item, no future info)."""
    df = sales_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["item_id", "date"]).reset_index(drop=True)
    g = df.groupby("item_id")["units_sold"]
    df["o_demand"] = df["units_sold"]
    df["o_lag6"] = g.shift(6)
    df["o_roll7"] = g.transform(lambda x: x.rolling(7, min_periods=1).mean())
    df["o_roll14"] = g.transform(lambda x: x.rolling(14, min_periods=1).mean())
    return df


def build_multistep_dataset(sales_df: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """Build the (origin, horizon) training table with a target and baseline."""
    df = _origin_features(sales_df)
    g = df.groupby("item_id")["units_sold"]

    holiday = (
        df.drop_duplicates("date").set_index("date")["is_holiday"]
        if "is_holiday" in df.columns
        else None
    )
    demand_lookup = df.set_index(["item_id", "date"])["units_sold"]

    frames = []
    for h in horizons:
        t = df.copy()
        t["horizon"] = h
        t["target"] = g.shift(-h)                                  # label: demand at t+h
        t["target_date"] = t["date"] + pd.Timedelta(days=h)
        t["t_dow"] = t["target_date"].dt.dayofweek
        t["t_month"] = t["target_date"].dt.month
        t["t_is_weekend"] = (t["t_dow"] >= 5).astype(int)
        if holiday is not None:
            t["t_is_holiday"] = t["target_date"].map(holiday).fillna(0).astype(int)
        else:
            t["t_is_holiday"] = 0
        # Seasonal-naive baseline: demand on the same weekday one week before the target.
        keys = list(zip(t["item_id"], t["target_date"] - pd.Timedelta(days=7), strict=True))
        t["seasonal_naive"] = demand_lookup.reindex(keys).to_numpy()
        frames.append(t)

    out = pd.concat(frames, ignore_index=True)
    return out.dropna(subset=["target", "o_lag6"]).reset_index(drop=True)


def _encode(out: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict]:
    """One-hot the category, integer-encode the item; return (df, features, codes)."""
    if "category" in out.columns:
        out = pd.get_dummies(out, columns=["category"], drop_first=True)
    codes = {item: i for i, item in enumerate(sorted(out["item_id"].unique()))}
    out["item_id_encoded"] = out["item_id"].map(codes)
    features = BASE_FEATURES + [c for c in out.columns if c.startswith("category_")] + ["item_id_encoded"]
    return out, features, codes


def forecast_next_days(sales_df: pd.DataFrame, item_id: str, bundle: dict) -> pd.DataFrame:
    """Forecast the next ``len(horizons)`` days for one item from its latest data."""
    hist = sales_df[sales_df["item_id"] == item_id].sort_values("date")
    hist = hist.assign(date=pd.to_datetime(hist["date"]))
    units = hist["units_sold"].to_numpy()
    origin_date = hist["date"].iloc[-1]
    category = hist["category"].iloc[-1] if "category" in hist.columns else None

    base = {
        "o_demand": units[-1],
        "o_lag6": units[-7] if len(units) >= 7 else units[0],
        "o_roll7": units[-7:].mean(),
        "o_roll14": units[-14:].mean(),
        "item_id_encoded": bundle["item_codes"].get(item_id, -1),
    }

    rows = []
    for h in bundle["horizons"]:
        target_date = origin_date + pd.Timedelta(days=h)
        row = dict(base)
        row.update({
            "horizon": h,
            "t_dow": target_date.dayofweek,
            "t_month": target_date.month,
            "t_is_weekend": int(target_date.dayofweek >= 5),
            "t_is_holiday": 0,  # future holiday calendar not assumed at inference
            "target_date": target_date,
        })
        if category is not None:
            row[f"category_{category}"] = 1
        rows.append(row)

    frame = pd.DataFrame(rows).reindex(columns=bundle["features"] + ["target_date"], fill_value=0)
    frame["predicted_demand"] = np.clip(bundle["model"].predict(frame[bundle["features"]]), 0, None)
    return frame[["target_date", "predicted_demand"]]


def main(source: str = "synthetic") -> None:
    config = get_config(source)
    print(f"Building 7-day-ahead dataset for '{source}'...")
    sales = pd.read_csv(resolve(config["sales"]))
    sales["date"] = pd.to_datetime(sales["date"])

    data = build_multistep_dataset(sales)
    data, features, codes = _encode(data)

    # Split by ORIGIN date so we never train on origins from the test period.
    cutoff = get_test_cutoff(sales, config["split_date"], config["test_fraction"])
    train = data[data["date"] < cutoff]
    test = data[data["date"] >= cutoff]
    print(f"Train rows: {len(train):,} | Test rows: {len(test):,} | Features: {len(features)}")

    model = HistGradientBoostingRegressor(**HGB_PARAMS)
    model.fit(train[features], train["target"])
    test = test.assign(pred=model.predict(test[features]))

    # MAE by horizon: model vs seasonal-naive baseline.
    print(f"\n{'horizon':>7} {'model_MAE':>10} {'naive_MAE':>10}")
    mae_by_h, base_by_h = {}, {}
    for h in HORIZONS:
        sub = test[test["horizon"] == h]
        mae_by_h[h] = float(mean_absolute_error(sub["target"], sub["pred"]))
        base = sub.dropna(subset=["seasonal_naive"])
        base_by_h[h] = float(mean_absolute_error(base["target"], base["seasonal_naive"]))
        print(f"{h:>7} {mae_by_h[h]:>10.2f} {base_by_h[h]:>10.2f}")

    overall = mean_absolute_error(test["target"], test["pred"])
    naive = test.dropna(subset=["seasonal_naive"])
    overall_naive = mean_absolute_error(naive["target"], naive["seasonal_naive"])
    print(f"\nOverall MAE: model {overall:.2f} vs seasonal-naive {overall_naive:.2f} "
          f"({(overall_naive - overall) / overall_naive * 100:.1f}% better)")

    bundle = {
        "model": model,
        "features": features,
        "item_codes": codes,
        "horizons": HORIZONS,
        "source": source,
        "mae_by_horizon": mae_by_h,
        "baseline_mae_by_horizon": base_by_h,
        "overall_mae": float(overall),
        "overall_baseline_mae": float(overall_naive),
    }
    model_path = resolve(config["multistep_model"])
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(bundle, model_path, compress=3)
    print(f"Saved {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the multi-step (7-day) forecaster.")
    parser.add_argument("--source", choices=["synthetic", "real"], default="synthetic")
    args = parser.parse_args()
    main(source=args.source)
