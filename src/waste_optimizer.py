"""
Waste-optimization module — the "business value" layer.

Converts raw demand predictions into an actionable prep recommendation and
quantifies how much food waste that saves versus a naive over-prep strategy.

Strategy comparison:
- **Baseline**: prep the maximum quantity sold over the last 14 days. This is
  what a cautious manager does by hand to avoid ever running out.
- **ML**: prep the model's predicted demand plus a small safety margin.

Waste = prepared but unsold. Stockout = demand that exceeded prep.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_prep_quantity(predictions, safety_margin: float = 0.10) -> np.ndarray:
    """Recommended units to prepare: ``ceil(predicted * (1 + safety_margin))``.

    We round *up* because you cannot prepare a fraction of a dish and want to
    stay on the safe side of demand. ``np.round`` first strips floating-point
    error (e.g. ``50 * 1.1 == 55.00000000000001``) so exact values do not get
    bumped to the next integer.

    Args:
        predictions: Predicted demand per row.
        safety_margin: Fractional buffer added on top of the prediction.

    Returns:
        Integer-valued array of recommended prep quantities.
    """
    padded = np.asarray(predictions, dtype=float) * (1.0 + safety_margin)
    return np.ceil(np.round(padded, 6))


def simulate_waste(actual_demand, prep_quantity) -> tuple[np.ndarray, np.ndarray]:
    """Return per-row ``(waste, stockout)`` given actuals and prepared amounts."""
    actual = np.asarray(actual_demand, dtype=float)
    prep = np.asarray(prep_quantity, dtype=float)

    waste = np.maximum(0.0, prep - actual)      # prepared but not sold
    stockout = np.maximum(0.0, actual - prep)   # demand we could not meet
    return waste, stockout


def evaluate_waste_reduction(
    df: pd.DataFrame,
    predicted_col: str = "predicted_demand",
    actual_col: str = "units_sold",
    safety_margin: float = 0.10,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compare ML-driven prep against the naive 14-day-max baseline.

    Args:
        df: Frame with ``item_id``, ``date``, the actual and predicted columns.
        predicted_col: Column holding model predictions.
        actual_col: Column holding realised demand.
        safety_margin: Buffer applied to the ML prep quantity.

    Returns:
        The augmented dataframe (per-row prep/waste/stockout for both
        strategies) and an aggregate metrics dictionary.
    """
    df = df.sort_values(by=["item_id", "date"]).reset_index(drop=True)

    # Baseline prep = rolling 14-day max of prior demand (shift(1) excludes today).
    df["baseline_prep"] = df.groupby("item_id")[actual_col].transform(
        lambda x: x.shift(1).rolling(window=14, min_periods=1).max()
    )
    df["baseline_prep"] = df["baseline_prep"].fillna(df[actual_col])

    df["ml_prep"] = calculate_prep_quantity(df[predicted_col], safety_margin=safety_margin)

    df["ml_waste"], df["ml_stockout"] = simulate_waste(df[actual_col], df["ml_prep"])
    df["baseline_waste"], df["baseline_stockout"] = simulate_waste(
        df[actual_col], df["baseline_prep"]
    )

    total_ml_waste = float(df["ml_waste"].sum())
    total_baseline_waste = float(df["baseline_waste"].sum())

    waste_reduction_percent = 0.0
    if total_baseline_waste > 0:
        waste_reduction_percent = (
            (total_baseline_waste - total_ml_waste) / total_baseline_waste
        ) * 100

    metrics = {
        "total_ml_waste_units": total_ml_waste,
        "total_baseline_waste_units": total_baseline_waste,
        "waste_reduction_percent": waste_reduction_percent,
        "total_ml_stockout_units": float(df["ml_stockout"].sum()),
        "total_baseline_stockout_units": float(df["baseline_stockout"].sum()),
    }
    return df, metrics


# --------------------------------------------------------------------------- #
# Cost-aware optimization (INR)
# --------------------------------------------------------------------------- #
# Two mistakes cost money in different ways:
#   * Waste   -> you already paid the ``food_cost`` for units nobody bought.
#   * Stockout -> you forgo the *profit margin* (price - food_cost) on demand
#                 you could not serve.
# Minimising the *sum* of these rupee costs is what a manager actually cares
# about, and it is what picks the "right" safety margin.


def add_cost_columns(
    df: pd.DataFrame,
    waste_col: str,
    stockout_col: str,
    prefix: str,
) -> pd.DataFrame:
    """Attach per-row INR waste/stockout/total cost for one strategy.

    ``df`` must already carry ``selling_price`` and ``food_cost`` columns.

    Args:
        df: Frame with the given waste/stockout + cost columns.
        waste_col: Column holding wasted units.
        stockout_col: Column holding stockout units.
        prefix: Prefix for the created cost columns (e.g. ``"ml"``).

    Returns:
        ``df`` with ``{prefix}_waste_cost``, ``{prefix}_stockout_cost`` and
        ``{prefix}_total_cost`` columns (all in INR).
    """
    margin = df["selling_price"] - df["food_cost"]
    df[f"{prefix}_waste_cost"] = df[waste_col] * df["food_cost"]
    df[f"{prefix}_stockout_cost"] = df[stockout_col] * margin
    df[f"{prefix}_total_cost"] = df[f"{prefix}_waste_cost"] + df[f"{prefix}_stockout_cost"]
    return df


def evaluate_cost_impact(
    eval_df: pd.DataFrame,
    menu_df: pd.DataFrame,
) -> dict[str, float]:
    """Total INR cost of the ML vs baseline strategy on an evaluated frame.

    ``eval_df`` must already contain the waste/stockout columns produced by
    :func:`evaluate_waste_reduction` (``ml_waste``, ``ml_stockout``,
    ``baseline_waste``, ``baseline_stockout``).
    """
    # Merge item economics once, then cost both strategies.
    df = eval_df.merge(
        menu_df[["item_id", "selling_price", "food_cost"]], on="item_id", how="left"
    )
    df = add_cost_columns(df, "ml_waste", "ml_stockout", "ml")
    df = add_cost_columns(df, "baseline_waste", "baseline_stockout", "baseline")

    ml_total = float(df["ml_total_cost"].sum())
    base_total = float(df["baseline_total_cost"].sum())
    savings = base_total - ml_total

    return {
        "ml_waste_cost": float(df["ml_waste_cost"].sum()),
        "ml_stockout_cost": float(df["ml_stockout_cost"].sum()),
        "ml_total_cost": ml_total,
        "baseline_waste_cost": float(df["baseline_waste_cost"].sum()),
        "baseline_stockout_cost": float(df["baseline_stockout_cost"].sum()),
        "baseline_total_cost": base_total,
        "cost_savings": savings,
        "cost_savings_percent": (savings / base_total * 100) if base_total > 0 else 0.0,
    }


def find_optimal_margin(
    scored_df: pd.DataFrame,
    menu_df: pd.DataFrame,
    margins=None,
) -> tuple[float, pd.DataFrame]:
    """Search safety margins for the one that minimises total ML INR cost.

    Args:
        scored_df: Frame with ``item_id``, ``date``, ``units_sold`` and a
            ``predicted_demand`` column.
        menu_df: Menu table with INR cost columns.
        margins: Iterable of safety margins to try (defaults to 0%..50%).

    Returns:
        The cost-minimising margin and a per-margin summary dataframe
        (margin, ml/baseline cost, savings, waste-reduction %).
    """
    if margins is None:
        margins = np.arange(0.0, 0.51, 0.05)

    rows = []
    for margin in margins:
        eval_df, waste_metrics = evaluate_waste_reduction(scored_df, safety_margin=float(margin))
        cost = evaluate_cost_impact(eval_df, menu_df)
        rows.append(
            {
                "safety_margin": round(float(margin), 2),
                "ml_total_cost": cost["ml_total_cost"],
                "baseline_total_cost": cost["baseline_total_cost"],
                "cost_savings": cost["cost_savings"],
                "cost_savings_percent": cost["cost_savings_percent"],
                "waste_reduction_percent": waste_metrics["waste_reduction_percent"],
                "ml_stockout_units": waste_metrics["total_ml_stockout_units"],
            }
        )

    summary = pd.DataFrame(rows)
    best_margin = float(summary.loc[summary["ml_total_cost"].idxmin(), "safety_margin"])
    return best_margin, summary


def evaluate_prep_strategy(
    df: pd.DataFrame,
    menu_df: pd.DataFrame,
    prep_col: str,
    actual_col: str = "units_sold",
) -> dict[str, float]:
    """Score an arbitrary prep column (e.g. a quantile forecast) in units + INR.

    Returns the achieved **service level** (fraction of rows where prep met
    demand), total wasted/stockout units, and the total INR cost — so a
    quantile-based prep can be compared apples-to-apples with the margin-based one.
    """
    waste, stockout = simulate_waste(df[actual_col], df[prep_col])
    scored = df.assign(_waste=waste, _stockout=stockout).merge(
        menu_df[["item_id", "selling_price", "food_cost"]], on="item_id", how="left"
    )
    margin = scored["selling_price"] - scored["food_cost"]
    waste_cost = float((scored["_waste"] * scored["food_cost"]).sum())
    stockout_cost = float((scored["_stockout"] * margin).sum())

    return {
        "service_level": float((df[actual_col] <= df[prep_col]).mean()),
        "waste_units": float(waste.sum()),
        "stockout_units": float(stockout.sum()),
        "waste_cost": waste_cost,
        "stockout_cost": stockout_cost,
        "total_cost": waste_cost + stockout_cost,
    }
