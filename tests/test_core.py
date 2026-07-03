"""
Unit tests for the core pipeline: data generation, feature engineering,
prediction, and the waste optimizer.

Run with: ``pytest tests/ -q``
"""

import os
import sys

import numpy as np
import pandas as pd

# Make ``src`` importable when tests run from the repo root.
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

import prepare_real_data  # noqa: E402
from data_generation import (  # noqa: E402
    START_DATE,
    generate_menu_items,
    generate_sales_data,
    generate_weather,
)
from features import create_features, get_test_cutoff, train_test_split_by_date  # noqa: E402
from multistep import (  # noqa: E402
    BASE_FEATURES,
    _encode,
    build_multistep_dataset,
    forecast_next_days,
)
from predict import predict_demand  # noqa: E402
from quantile_model import pinball_loss, predict_quantiles  # noqa: E402
from waste_optimizer import (  # noqa: E402
    calculate_prep_quantity,
    evaluate_cost_impact,
    evaluate_prep_strategy,
    evaluate_waste_reduction,
    find_optimal_margin,
    simulate_waste,
)


# --------------------------------------------------------------------------- #
# Data generation
# --------------------------------------------------------------------------- #
def test_generate_menu_items():
    df = generate_menu_items()
    assert not df.empty
    assert {"item_id", "category", "name", "selling_price", "food_cost"}.issubset(df.columns)
    assert len(df) >= 15  # ~20 items configured
    assert df["item_id"].is_unique
    # Food cost (COGS) must be below the selling price for every item.
    assert (df["food_cost"] < df["selling_price"]).all()


def test_generate_weather_seasonal_range():
    dates = pd.date_range("2023-01-01", "2023-12-31")
    temp = generate_weather(dates)
    assert len(temp) == 365
    # Synthetic temperatures should stay in a believable band.
    assert temp.min() > -20
    assert temp.max() < 120


def test_sales_data_is_non_negative_and_shaped():
    menu = generate_menu_items()
    sales = generate_sales_data(menu, START_DATE, pd.Timestamp("2021-06-30"))
    assert (sales["units_sold"] >= 0).all()          # demand is never negative
    assert sales["units_sold"].dtype.kind in "iu"    # integer counts (Poisson)
    # One row per item per day.
    n_days = (pd.Timestamp("2021-06-30") - START_DATE).days + 1
    assert len(sales) == len(menu) * n_days


def test_weekend_demand_exceeds_weekday():
    """The generator bakes in weekly seasonality; verify it actually shows up."""
    menu = generate_menu_items()
    sales = generate_sales_data(menu, START_DATE, pd.Timestamp("2021-12-31"))
    weekend = sales[sales["is_weekend"] == 1]["units_sold"].mean()
    weekday = sales[sales["is_weekend"] == 0]["units_sold"].mean()
    assert weekend > weekday


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def _toy_frame(n=20):
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=n),
            "item_id": ["ITEM_001"] * n,
            "units_sold": np.arange(10, 10 + n),  # deterministic, easy to check
        }
    )


def test_create_features_adds_expected_columns():
    out = create_features(_toy_frame())
    for col in ["lag_1", "lag_7", "lag_14", "rolling_mean_7", "day_of_week", "month", "trend_index"]:
        assert col in out.columns


def test_create_features_drops_undefined_leading_rows():
    # 20 rows, longest lag is 14 -> first 14 rows dropped, 6 remain.
    out = create_features(_toy_frame(20))
    assert len(out) == 6


def test_lag_features_have_no_future_leakage():
    """lag_1 on a given date must equal the *previous* day's actual demand."""
    toy = _toy_frame(20)
    out = create_features(toy).reset_index(drop=True)
    merged = out.merge(
        toy.rename(columns={"units_sold": "actual"}), on=["date", "item_id"]
    )
    # units_sold increments by 1 each day, so yesterday's value == today - 1.
    assert (merged["lag_1"] == merged["actual"] - 1).all()


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
class _StubModel:
    """Minimal model whose predictions include a negative value."""

    def predict(self, X):
        return np.array([-5.0, 0.0, 12.3][: len(X)])


def test_predict_demand_shape_and_non_negative():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    info = {"model": _StubModel(), "features": ["a", "b"]}
    preds = predict_demand(df, info)
    assert len(preds) == len(df)
    assert (preds >= 0).all()  # negatives are clipped to 0


def test_predict_demand_raises_on_missing_feature():
    df = pd.DataFrame({"a": [1, 2, 3]})
    info = {"model": _StubModel(), "features": ["a", "b"]}
    try:
        predict_demand(df, info)
        raise AssertionError("expected ValueError for missing feature")
    except ValueError:
        pass


# --------------------------------------------------------------------------- #
# Waste optimizer
# --------------------------------------------------------------------------- #
def test_calculate_prep_quantity_exact_values():
    # 10% margin: 100->110, 50->55, 0->0 (no floating-point over-shoot).
    prep = calculate_prep_quantity([100, 50, 0], safety_margin=0.10)
    assert list(prep) == [110.0, 55.0, 0.0]


def test_prep_quantity_never_below_prediction():
    preds = np.array([3, 7, 21, 44])
    prep = calculate_prep_quantity(preds, safety_margin=0.15)
    assert (prep >= preds).all()


def test_simulate_waste():
    waste, stockout = simulate_waste([100, 60, 40], [110, 50, 40])
    assert list(waste) == [10.0, 0.0, 0.0]
    assert list(stockout) == [0.0, 10.0, 0.0]


def test_evaluate_waste_reduction_metrics():
    df = pd.DataFrame(
        {
            "item_id": ["ITEM_001"] * 5,
            "date": pd.date_range("2023-07-01", periods=5),
            "units_sold": [10, 12, 11, 9, 13],
            "predicted_demand": [10, 11, 11, 10, 12],
        }
    )
    out, metrics = evaluate_waste_reduction(df, safety_margin=0.10)
    expected_keys = {
        "total_ml_waste_units",
        "total_baseline_waste_units",
        "waste_reduction_percent",
        "total_ml_stockout_units",
        "total_baseline_stockout_units",
    }
    assert expected_keys.issubset(metrics.keys())
    assert {"ml_prep", "baseline_prep", "ml_waste", "baseline_waste"}.issubset(out.columns)
    # The reported reduction % must be consistent with the underlying waste totals.
    base, ml = metrics["total_baseline_waste_units"], metrics["total_ml_waste_units"]
    expected_pct = (base - ml) / base * 100 if base > 0 else 0.0
    assert metrics["waste_reduction_percent"] == expected_pct


# --------------------------------------------------------------------------- #
# INR cost model
# --------------------------------------------------------------------------- #
def _cost_frames():
    df = pd.DataFrame(
        {
            "item_id": ["ITEM_001"] * 6,
            "date": pd.date_range("2023-07-01", periods=6),
            "units_sold": [10, 12, 11, 9, 13, 10],
            "predicted_demand": [10, 11, 11, 10, 12, 10],
        }
    )
    menu = pd.DataFrame(
        {"item_id": ["ITEM_001"], "selling_price": [500.0], "food_cost": [175.0]}
    )
    return df, menu


def test_cost_impact_totals_add_up():
    df, menu = _cost_frames()
    eval_df, _ = evaluate_waste_reduction(df, safety_margin=0.10)
    cost = evaluate_cost_impact(eval_df, menu)
    # Total = waste cost + stockout cost, for each strategy.
    assert cost["ml_total_cost"] == cost["ml_waste_cost"] + cost["ml_stockout_cost"]
    assert cost["baseline_total_cost"] == (
        cost["baseline_waste_cost"] + cost["baseline_stockout_cost"]
    )
    # Waste is priced at food cost; stockout at the profit margin.
    assert cost["ml_waste_cost"] >= 0 and cost["ml_stockout_cost"] >= 0


def test_find_optimal_margin_minimises_cost():
    df, menu = _cost_frames()
    best_margin, summary = find_optimal_margin(df, menu, margins=[0.0, 0.1, 0.2, 0.3])
    # The returned margin must be the one with the lowest ML total cost.
    assert best_margin == summary.loc[summary["ml_total_cost"].idxmin(), "safety_margin"]
    assert set(summary["safety_margin"]) == {0.0, 0.1, 0.2, 0.3}


# --------------------------------------------------------------------------- #
# Chronological split
# --------------------------------------------------------------------------- #
def test_split_by_fraction_is_chronological_and_non_overlapping():
    df = pd.DataFrame(
        {"date": pd.date_range("2023-01-01", periods=100), "item_id": "A", "units_sold": 1}
    )
    train, test = train_test_split_by_date(df, test_fraction=0.2)
    assert len(train) == 80 and len(test) == 20
    # No leakage: every training date is strictly before every test date.
    assert train["date"].max() < test["date"].min()


def test_explicit_split_date_takes_precedence():
    df = pd.DataFrame(
        {"date": pd.date_range("2023-01-01", periods=10), "item_id": "A", "units_sold": 1}
    )
    assert get_test_cutoff(df, split_date="2023-01-05") == pd.Timestamp("2023-01-05")


# --------------------------------------------------------------------------- #
# Real-data mapping (offline — no network, uses crafted raw frames)
# --------------------------------------------------------------------------- #
def test_build_canonical_maps_recruit_schema(monkeypatch):
    # Relax the subsampling thresholds so tiny fixtures survive.
    monkeypatch.setattr(prepare_real_data, "MIN_DAYS", 2)
    monkeypatch.setattr(prepare_real_data, "N_STORES", 5)

    visits = pd.DataFrame(
        {
            "air_store_id": ["air_a", "air_a", "air_a", "air_b", "air_b", "air_b"],
            "visit_date": ["2016-01-01", "2016-01-02", "2016-01-03"] * 2,
            "visitors": [10, 20, 15, 5, 8, 6],
        }
    )
    dates = pd.DataFrame(
        {
            "calendar_date": ["2016-01-01", "2016-01-02", "2016-01-03"],
            "day_of_week": ["Friday", "Saturday", "Sunday"],
            "holiday_flg": [1, 0, 0],
        }
    )
    stores = pd.DataFrame(
        {
            "air_store_id": ["air_a", "air_b"],
            "air_genre_name": ["Izakaya", "Cafe/Sweets"],
            "air_area_name": ["Tokyo", "Osaka"],
        }
    )

    sales, menu = prepare_real_data.build_canonical(visits, dates, stores)
    assert {"date", "item_id", "category", "units_sold", "is_holiday"} == set(sales.columns)
    assert {"item_id", "category", "name", "selling_price", "food_cost"} == set(menu.columns)
    assert sales["is_holiday"].isin([0, 1]).all()
    # INR economics are populated and internally consistent.
    assert (menu["food_cost"] < menu["selling_price"]).all()
    assert (menu["selling_price"] > 0).all()


# --------------------------------------------------------------------------- #
# Prediction intervals (quantile regression)
# --------------------------------------------------------------------------- #
class _FixedQuantileModel:
    """Returns a constant prediction — lets us test the cross-fix deterministically."""

    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


def test_predict_quantiles_are_non_crossing_and_non_negative():
    # Deliberately give the higher quantile a LOWER raw prediction to force a crossing.
    bundle = {
        "features": ["a"],
        "quantiles": [0.1, 0.9],
        "models": {0.1: _FixedQuantileModel(8.0), 0.9: _FixedQuantileModel(-3.0)},
    }
    df = pd.DataFrame({"a": [1, 2, 3]})
    out = predict_quantiles(df, bundle)
    assert out[0.1].shape == (3,)
    # Clipped at 0 and re-sorted so P10 <= P90 on every row.
    assert (out[0.9] >= out[0.1]).all()
    assert (out[0.1] >= 0).all() and (out[0.9] >= 0).all()


def test_pinball_loss_reduces_to_half_abs_error_at_median():
    # For q=0.5 the pinball loss is 0.5 * mean(|error|).
    assert pinball_loss([10, 20], [8, 24], 0.5) == 0.5 * np.mean([2, 4])


def test_evaluate_prep_strategy_service_level_and_cost():
    df = pd.DataFrame({
        "item_id": ["ITEM_001"] * 4,
        "units_sold": [10, 10, 10, 10],
        "prep": [12, 9, 10, 15],  # meets demand on 3 of 4 rows
    })
    menu = pd.DataFrame({"item_id": ["ITEM_001"], "selling_price": [500.0], "food_cost": [175.0]})
    m = evaluate_prep_strategy(df, menu, "prep")
    assert m["service_level"] == 0.75          # prep >= demand on 3/4 rows
    assert m["waste_units"] == 2 + 0 + 0 + 5   # over-prep on rows 1 and 4
    assert m["stockout_units"] == 1            # under-prep of 1 on row 2
    assert m["total_cost"] == m["waste_cost"] + m["stockout_cost"]


# --------------------------------------------------------------------------- #
# Multi-step (7-day) forecasting
# --------------------------------------------------------------------------- #
def _multi_series(days=40):
    # Single item whose demand increments by 1 each day -> easy to reason about.
    return pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=days),
        "item_id": ["ITEM_001"] * days,
        "units_sold": range(10, 10 + days),
        "is_holiday": 0,
    })


def test_build_multistep_dataset_targets_and_no_leakage():
    ds = build_multistep_dataset(_multi_series(40), horizons=[1, 7])
    assert {"target", "horizon", "o_demand", "o_lag6", "t_dow", "t_is_holiday"}.issubset(ds.columns)
    # Demand increments by 1/day, so target at horizon h is o_demand + h,
    # and o_lag6 (6 days before the origin) is o_demand - 6.
    for _, row in ds.iterrows():
        assert row["target"] == row["o_demand"] + row["horizon"]
        assert row["o_lag6"] == row["o_demand"] - 6


def test_encode_produces_horizon_and_item_features():
    ds = build_multistep_dataset(_multi_series(30), horizons=[1, 2, 3])
    encoded, features, codes = _encode(ds)
    assert set(BASE_FEATURES).issubset(features)
    assert "item_id_encoded" in features
    assert codes["ITEM_001"] == 0


def test_forecast_next_days_returns_full_horizon():
    ds = build_multistep_dataset(_multi_series(30), horizons=[1, 2, 3])
    _, features, codes = _encode(ds)
    bundle = {
        "model": _FixedQuantileModel(42.0),
        "features": features,
        "item_codes": codes,
        "horizons": [1, 2, 3],
    }
    fc = forecast_next_days(_multi_series(30), "ITEM_001", bundle)
    assert list(fc.columns) == ["target_date", "predicted_demand"]
    assert len(fc) == 3
    # Forecast dates are strictly increasing and start the day after the last origin.
    assert fc["target_date"].is_monotonic_increasing
    assert fc["target_date"].iloc[0] == pd.Timestamp("2023-01-30") + pd.Timedelta(days=1)
