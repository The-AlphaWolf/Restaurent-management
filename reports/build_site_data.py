"""
Build ``docs/data.json`` — the data behind the static GitHub Pages site.

Everything the site shows is computed here from the committed datasets and the
trained model bundles, so the page never fakes a number: the same scoring path
as ``generate_report.py`` (held-out test period only).

Run: ``python reports/build_site_data.py``
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd

REPORTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(REPORTS_DIR)
sys.path.insert(0, REPORTS_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from generate_report import _load_optional, _score_test  # noqa: E402

from datasets import DATASETS, get_config, resolve  # noqa: E402
from multistep import forecast_next_days  # noqa: E402
from waste_optimizer import evaluate_waste_reduction, find_optimal_margin  # noqa: E402

CHART_DAYS = 120  # how much of the test period the line chart shows


def _series(test: pd.DataFrame) -> list[dict]:
    daily = (
        test.groupby("date")[["units_sold", "predicted_demand"]].sum().tail(CHART_DAYS).round(1)
    )
    return [
        {"date": d.strftime("%Y-%m-%d"), "actual": float(a), "predicted": float(p)}
        for d, a, p in zip(daily.index, daily["units_sold"], daily["predicted_demand"], strict=True)
    ]


def _forecast(source: str) -> list[dict]:
    """Sum the 7-day-ahead forecast across every menu item (empty if untrained)."""
    bundle = _load_optional(source, "multistep_model")
    if bundle is None:
        return []
    sales = pd.read_csv(resolve(get_config(source)["sales"]))
    sales["date"] = pd.to_datetime(sales["date"])
    total = None
    for item in sales["item_id"].unique():
        fc = forecast_next_days(sales, item, bundle).set_index("target_date")["predicted_demand"]
        total = fc if total is None else total.add(fc, fill_value=0)
    return [
        {"date": d.strftime("%Y-%m-%d"), "predicted": round(float(v), 1)}
        for d, v in total.items()
    ]


def _heatmap(test: pd.DataFrame, sales: pd.DataFrame) -> dict:
    """Mean units/day per (category, weekday) over the test period."""
    cats = sales[["item_id", "category"]].drop_duplicates() if "category" in sales else None
    if cats is None:
        return {"rows": [], "cols": [], "values": []}
    df = test[["date", "item_id", "units_sold"]].merge(cats, on="item_id", how="left")
    df["dow"] = df["date"].dt.dayofweek
    grid = df.groupby(["category", "dow"])["units_sold"].mean().unstack(fill_value=0.0)
    grid = grid.reindex(columns=range(7), fill_value=0.0)
    # Busiest categories first, capped so the grid stays legible.
    grid = grid.loc[grid.sum(axis=1).sort_values(ascending=False).index][:8]
    return {
        "rows": [str(c) for c in grid.index],
        "cols": ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"],
        "values": grid.round(1).to_numpy().tolist(),
    }


def _waste_by_month(test: pd.DataFrame, margin: float) -> list[dict]:
    """Waste reduction vs the 14-day-max baseline, month by month."""
    out = []
    for period, chunk in test.groupby(test["date"].dt.to_period("M")):
        _, metrics = evaluate_waste_reduction(chunk, safety_margin=margin)
        out.append(
            {
                "month": str(period),
                "reduction": round(metrics["waste_reduction_percent"], 1),
            }
        )
    return out


def build(source: str) -> dict | None:
    scored = _score_test(source)
    if scored is None:
        return None
    test, menu, bundle = scored
    sales = pd.read_csv(resolve(get_config(source)["sales"]))
    sales["date"] = pd.to_datetime(sales["date"])

    best_margin, summary = find_optimal_margin(test, menu)
    best = summary[summary["safety_margin"] == best_margin].iloc[0]
    _, waste_metrics = evaluate_waste_reduction(test, safety_margin=best_margin)

    m = bundle["metrics"]
    chosen = m["gradient_boosting"] if bundle["model_name"] == "Gradient Boosting" else m["random_forest"]
    multistep = _load_optional(source, "multistep_model")

    return {
        "label": get_config(source)["label"],
        "model": bundle["model_name"],
        "metrics": {
            "mae": round(chosen["mae"], 2),
            "rmse": round(chosen["rmse"], 2),
            "mape": round(chosen["mape"], 1),
            "r2": round(chosen["r2"], 3),
            "baseline_mae": round(m["baseline"]["mae"], 2),
            "mae_gain_pct": round((m["baseline"]["mae"] - chosen["mae"]) / m["baseline"]["mae"] * 100, 1),
            "waste_reduction_pct": round(waste_metrics["waste_reduction_percent"], 1),
            "cost_saving_pct": round(float(best["cost_savings_percent"]), 1),
            "baseline_cost": float(best["baseline_total_cost"]),
            "ml_cost": float(best["ml_total_cost"]),
            "optimal_margin": best_margin,
            "horizon_days": max(multistep["horizons"]) if multistep else 1,
            "horizon_gain_pct": round(
                (multistep["overall_baseline_mae"] - multistep["overall_mae"])
                / multistep["overall_baseline_mae"] * 100, 1
            ) if multistep else None,
        },
        "series": _series(test),
        "forecast": _forecast(source),
        "heatmap": _heatmap(test, sales),
        "waste_by_month": _waste_by_month(test, best_margin),
        "top_features": [name for name, _ in bundle["feature_importance"][:4]],
        "facts": {
            "days": int(sales["date"].nunique()),
            "items": int(sales["item_id"].nunique()),
            "features": len(bundle["features"]),
            "rows": int(len(sales)),
            "test_from": test["date"].min().strftime("%Y-%m-%d"),
            "test_to": test["date"].max().strftime("%Y-%m-%d"),
        },
    }


def main() -> None:
    payload = {src: data for src in DATASETS if (data := build(src))}
    if not payload:
        print("No trained models found. Run training first.")
        return
    out_path = resolve("docs/data.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    print(f"Wrote {out_path} for: {', '.join(payload)}")


if __name__ == "__main__":
    main()
