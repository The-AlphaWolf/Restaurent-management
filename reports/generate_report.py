"""
Generate the results report.

For every available dataset (synthetic + real) this script:
1. Loads the trained model and scores the held-out test period.
2. Saves four figures to ``reports/figures/``:
     - actual vs predicted demand,
     - permutation feature importance,
     - waste-reduction / stockout tradeoff across safety margins,
     - total INR cost vs safety margin (with the cost-optimal point marked).
3. Writes ``RESULTS.md`` embedding those figures with the headline numbers.

Run: ``python reports/generate_report.py``
"""

from __future__ import annotations

import os
import sys

import joblib
import matplotlib
import pandas as pd

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt  # noqa: E402

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from datasets import DATASETS, get_config, resolve  # noqa: E402
from features import get_test_cutoff, preprocess_data  # noqa: E402
from predict import predict_demand  # noqa: E402
from waste_optimizer import evaluate_waste_reduction, find_optimal_margin  # noqa: E402

FIG_DIR = resolve("reports/figures")


def _score_test(source: str):
    """Return (test_data_with_predictions, menu_df, model_bundle) or None."""
    cfg = get_config(source)
    if not (os.path.exists(resolve(cfg["sales"])) and os.path.exists(resolve(cfg["model"]))):
        return None
    bundle = joblib.load(resolve(cfg["model"]))
    processed = preprocess_data(resolve(cfg["sales"]))
    cutoff = get_test_cutoff(processed, cfg["split_date"], cfg["test_fraction"])
    test = processed[processed["date"] >= cutoff].copy()
    test["predicted_demand"] = predict_demand(test, bundle)
    menu = pd.read_csv(resolve(cfg["menu"]))
    return test, menu, bundle


def _fig_actual_vs_predicted(test: pd.DataFrame, source: str) -> str:
    daily = test.groupby("date")[["units_sold", "predicted_demand"]].sum()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(daily.index, daily["units_sold"], label="Actual", linewidth=1.5)
    ax.plot(daily.index, daily["predicted_demand"], label="Predicted", linestyle="--")
    ax.set(title=f"Actual vs Predicted Demand ({source})", xlabel="Date", ylabel="Total units/day")
    ax.legend()
    fig.autofmt_xdate()
    return _save(fig, f"{source}_actual_vs_predicted.png")


def _fig_importance(bundle: dict, source: str) -> str:
    top = bundle["feature_importance"][:12][::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([n for n, _ in top], [s for _, s in top], color="#FF4B4B")
    ax.set(title=f"Permutation Feature Importance ({source})", xlabel="Error increase when shuffled")
    fig.tight_layout()
    return _save(fig, f"{source}_feature_importance.png")


def _fig_tradeoff(summary: pd.DataFrame, source: str) -> str:
    fig, ax1 = plt.subplots(figsize=(8, 4))
    x = summary["safety_margin"] * 100
    ax1.plot(x, summary["waste_reduction_percent"], color="#2E8B57", marker="o", label="Waste reduction %")
    ax1.set(xlabel="Safety margin (%)", ylabel="Waste reduction vs baseline (%)")
    ax1.tick_params(axis="y", labelcolor="#2E8B57")
    ax2 = ax1.twinx()
    ax2.plot(x, summary["ml_stockout_units"], color="#B22222", marker="s", label="ML stockout units")
    ax2.set_ylabel("ML stockout (units)", color="#B22222")
    ax2.tick_params(axis="y", labelcolor="#B22222")
    ax1.set_title(f"Waste vs Stockout Tradeoff ({source})")
    fig.tight_layout()
    return _save(fig, f"{source}_tradeoff.png")


def _fig_cost(summary: pd.DataFrame, best_margin: float, source: str) -> str:
    base_cost = summary["baseline_total_cost"].iloc[0]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(summary["safety_margin"] * 100, summary["ml_total_cost"], marker="o", label="ML total cost")
    ax.axhline(base_cost, linestyle=":", color="grey", label="Baseline cost")
    ax.axvline(best_margin * 100, linestyle="--", color="green", label=f"Optimal {best_margin:.0%}")
    ax.set(title=f"Total Cost vs Safety Margin ({source})", xlabel="Safety margin (%)", ylabel="Total cost (₹)")
    ax.legend()
    fig.tight_layout()
    return _save(fig, f"{source}_cost_curve.png")


def _save(fig, filename: str) -> str:
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return f"reports/figures/{filename}"  # repo-relative path for markdown


def _metrics_table(bundle: dict) -> str:
    m = bundle["metrics"]
    rows = [
        ("Naive baseline", m["baseline"]),
        ("Random Forest", m["random_forest"]),
        ("Gradient Boosting", m["gradient_boosting"]),
    ]
    lines = ["| Model | MAE | RMSE | MAPE | R² |", "|---|---|---|---|---|"]
    for name, mm in rows:
        lines.append(f"| {name} | {mm['mae']:.2f} | {mm['rmse']:.2f} | {mm['mape']:.1f}% | {mm['r2']:.3f} |")
    return "\n".join(lines)


def build_section(source: str) -> str | None:
    scored = _score_test(source)
    if scored is None:
        return None
    test, menu, bundle = scored
    best_margin, summary = find_optimal_margin(test, menu)
    best = summary[summary["safety_margin"] == best_margin].iloc[0]

    f_pred = _fig_actual_vs_predicted(test, source)
    f_imp = _fig_importance(bundle, source)
    f_trade = _fig_tradeoff(summary, source)
    f_cost = _fig_cost(summary, best_margin, source)

    m = bundle["metrics"]
    mae_impr = (m["baseline"]["mae"] - min(m["random_forest"]["mae"], m["gradient_boosting"]["mae"]))
    mae_pct = mae_impr / m["baseline"]["mae"] * 100
    # Waste reduction at the same-service-level (matched-stockout) is captured by
    # the tradeoff curve; here we report the value at the cost-optimal margin.
    _, waste_metrics = evaluate_waste_reduction(test, safety_margin=best_margin)

    return f"""## {get_config(source)['label']}

**Selected model:** {bundle['model_name']} &nbsp;·&nbsp; **Forecast error cut vs naive baseline:** {mae_pct:.1f}% (MAE {m['baseline']['mae']:.2f} → {min(m['random_forest']['mae'], m['gradient_boosting']['mae']):.2f})

{_metrics_table(bundle)}

![Actual vs Predicted]({f_pred})

### What drives demand
![Feature importance]({f_imp})

### Waste vs stockout tradeoff
The safety-margin slider trades food waste against stockouts. Higher margins prep more, wasting more but rarely running out.

![Tradeoff]({f_trade})

### Cost-optimal safety margin (INR)
Minimising total rupee cost (waste ingredient cost + lost-margin on stockouts) selects a **{best_margin:.0%} safety margin**, cutting cost from **₹{best['baseline_total_cost']:,.0f}** (baseline) to **₹{best['ml_total_cost']:,.0f}** — a **{best['cost_savings_percent']:.0f}% saving** (₹{best['cost_savings']:,.0f}), with **{waste_metrics['waste_reduction_percent']:.0f}% less waste**.

![Cost curve]({f_cost})
"""


def main() -> None:
    sections = [s for s in (build_section(src) for src in DATASETS) if s]
    if not sections:
        print("No trained models found. Run data generation + training first.")
        return

    report = (
        "# 📊 Results Report\n\n"
        "_Auto-generated by `python reports/generate_report.py`. All figures are "
        "produced from the held-out test period of each dataset._\n\n"
        "The same pipeline is evaluated on a **synthetic 3-year simulation** and on "
        "**real daily restaurant visitor data** (Recruit Restaurant, Japan), showing "
        "the approach generalises beyond hand-crafted data.\n\n---\n\n"
        + "\n\n---\n\n".join(sections)
        + "\n"
    )
    out_path = resolve("RESULTS.md")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"Wrote {out_path} with {len(sections)} dataset section(s).")


if __name__ == "__main__":
    main()
