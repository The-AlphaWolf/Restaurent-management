import os
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add src to path to import project modules.
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from datasets import DATASETS, get_config, resolve
from features import get_test_cutoff, preprocess_data
from predict import predict_demand
from waste_optimizer import evaluate_cost_impact, evaluate_waste_reduction, find_optimal_margin

st.set_page_config(page_title="Smart Restaurant Food Demand Prediction", layout="wide", page_icon="🍔")


# --------------------------------------------------------------------------- #
# Cached data / model loaders (keyed by dataset source)
# --------------------------------------------------------------------------- #
@st.cache_data
def load_sales_menu(source: str):
    cfg = get_config(source)
    sales_path, menu_path = resolve(cfg['sales']), resolve(cfg['menu'])
    if not os.path.exists(sales_path):
        return None, None
    sales_df = pd.read_csv(sales_path)
    sales_df['date'] = pd.to_datetime(sales_df['date'])
    menu_df = pd.read_csv(menu_path)
    return sales_df, menu_df


@st.cache_resource
def load_model_bundle(source: str):
    model_path = resolve(get_config(source)['model'])
    return joblib.load(model_path) if os.path.exists(model_path) else None


@st.cache_data
def score_test_data(source: str):
    """Load processed data, hold out the test period, and attach predictions."""
    cfg = get_config(source)
    sales_path = resolve(cfg['sales'])
    if not os.path.exists(sales_path):
        return None
    model_info = load_model_bundle(source)
    if model_info is None:
        return None
    processed = preprocess_data(sales_path)
    cutoff = get_test_cutoff(processed, cfg['split_date'], cfg['test_fraction'])
    test_data = processed[processed['date'] >= cutoff].copy()
    test_data['predicted_demand'] = predict_demand(test_data, model_info)
    return test_data


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def render_overview(sales_df, menu_df):
    st.header("📈 Historical Sales Overview")
    df = sales_df.merge(menu_df[['item_id', 'name']], on='item_id')

    col1, col2 = st.columns(2)
    with col1:
        daily_total = df.groupby('date')['units_sold'].sum().reset_index()
        fig = px.line(daily_total, x='date', y='units_sold', title="Total Daily Demand Over Time")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        category_total = df.groupby('category')['units_sold'].sum().reset_index()
        fig2 = px.pie(category_total, values='units_sold', names='category', title="Demand by Category")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Filter by Item")
    selected_item = st.selectbox("Select an item", sorted(df['name'].unique()))
    item_data = df[df['name'] == selected_item]
    fig3 = px.line(item_data, x='date', y='units_sold', title=f"Daily Demand for {selected_item}")
    st.plotly_chart(fig3, use_container_width=True)


def render_predictions(test_data, menu_df):
    st.header("🔮 Demand Predictions")
    st.write("Forecasts from the trained ML model, driven by historical patterns, "
             "seasonality, and calendar effects.")

    if test_data is None:
        st.warning("Data or model not found. Please run the training pipeline.")
        return

    test_data = test_data.merge(menu_df[['item_id', 'name', 'category']], on='item_id')
    selected_category = st.selectbox("Select Category", ['All'] + sorted(test_data['category'].unique()))
    display_data = test_data if selected_category == 'All' else test_data[test_data['category'] == selected_category]

    daily_actual = display_data.groupby('date')['units_sold'].sum().reset_index()
    daily_pred = display_data.groupby('date')['predicted_demand'].sum().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily_actual['date'], y=daily_actual['units_sold'], mode='lines', name='Actual Demand'))
    fig.add_trace(go.Scatter(x=daily_pred['date'], y=daily_pred['predicted_demand'], mode='lines',
                             name='Predicted Demand', line=dict(dash='dash')))
    fig.update_layout(title="Actual vs Predicted Demand", xaxis_title="Date", yaxis_title="Units Sold")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Item-Level Forecasts")
    selected_item = st.selectbox("Select an item", sorted(display_data['name'].unique()), key='item_pred')
    item_pred_data = display_data[display_data['name'] == selected_item].tail(14)

    col1, col2 = st.columns([2, 1])
    with col1:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=item_pred_data['date'], y=item_pred_data['units_sold'], name='Actual'))
        fig2.add_trace(go.Bar(x=item_pred_data['date'], y=item_pred_data['predicted_demand'], name='Predicted'))
        fig2.update_layout(barmode='group', title=f"14-Day Forecast Window: {selected_item}")
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        st.write("Recent Forecast Accuracy")
        mae = np.mean(np.abs(item_pred_data['units_sold'] - item_pred_data['predicted_demand']))
        st.metric("Mean Absolute Error (Units)", f"{mae:.1f}")


def render_waste_insights(test_data, menu_df):
    st.header("♻️ Waste Reduction Optimizer")
    if test_data is None:
        st.warning("Data or model not found.")
        return

    safety_margin = st.slider("Safety Margin (%)", min_value=0, max_value=50, value=10, step=5) / 100.0
    eval_df, metrics = evaluate_waste_reduction(test_data, safety_margin=safety_margin)

    st.subheader("Impact of ML-Driven Preparation")
    col1, col2, col3 = st.columns(3)
    col1.metric("Baseline Waste", f"{metrics['total_baseline_waste_units']:,.0f} units")
    col2.metric("ML Waste", f"{metrics['total_ml_waste_units']:,.0f} units",
                f"-{metrics['total_baseline_waste_units'] - metrics['total_ml_waste_units']:,.0f} units",
                delta_color="inverse")
    col3.metric("Waste Reduction", f"{metrics['waste_reduction_percent']:.1f}%")

    st.info("**Baseline**: prep the maximum quantity sold over the prior 14 days (cautious over-prep). "
            "**ML**: prep the predicted demand plus your chosen safety margin.")

    eval_df = eval_df.merge(menu_df[['item_id', 'name']], on='item_id')
    item_waste = eval_df.groupby('name')[['baseline_waste', 'ml_waste']].sum().reset_index()
    item_waste['reduction'] = item_waste['baseline_waste'] - item_waste['ml_waste']
    item_waste = item_waste.sort_values(by='reduction', ascending=False).head(10)
    fig = px.bar(item_waste, x='reduction', y='name', orientation='h',
                 title="Top 10 Items for Waste Reduction", labels={'reduction': 'Units Saved', 'name': 'Item'})
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)


def render_cost_impact(test_data, menu_df):
    st.header("💰 Cost Impact (INR)")
    if test_data is None:
        st.warning("Data or model not found.")
        return

    st.write("Turning units into rupees. **Waste cost** = ingredient cost of unsold prep. "
             "**Stockout cost** = lost profit margin on demand we could not serve. "
             "The optimal safety margin is the one that minimises their sum.")

    best_margin, summary = find_optimal_margin(test_data, menu_df)

    # Headline metrics at the cost-optimal margin.
    best_row = summary[summary['safety_margin'] == best_margin].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Cost-Optimal Safety Margin", f"{best_margin:.0%}")
    c2.metric("Baseline Cost", f"₹{best_row['baseline_total_cost']:,.0f}")
    c3.metric("ML Cost", f"₹{best_row['ml_total_cost']:,.0f}",
              f"-₹{best_row['cost_savings']:,.0f} ({best_row['cost_savings_percent']:.0f}%)",
              delta_color="inverse")

    # Cost-vs-margin curve.
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=summary['safety_margin'] * 100, y=summary['ml_total_cost'],
                             mode='lines+markers', name='ML total cost'))
    fig.add_hline(y=best_row['baseline_total_cost'], line_dash="dot",
                  annotation_text="Baseline cost", line_color="grey")
    fig.add_vline(x=best_margin * 100, line_dash="dash", line_color="green",
                  annotation_text=f"Optimal {best_margin:.0%}")
    fig.update_layout(title="Total Cost vs Safety Margin",
                      xaxis_title="Safety Margin (%)", yaxis_title="Total Cost (₹)")
    st.plotly_chart(fig, use_container_width=True)

    # Cost breakdown at the optimal margin.
    eval_df, _ = evaluate_waste_reduction(test_data, safety_margin=best_margin)
    cost = evaluate_cost_impact(eval_df, menu_df)
    breakdown = pd.DataFrame({
        'Strategy': ['Baseline', 'ML'],
        'Waste Cost (₹)': [cost['baseline_waste_cost'], cost['ml_waste_cost']],
        'Stockout Cost (₹)': [cost['baseline_stockout_cost'], cost['ml_stockout_cost']],
        'Total Cost (₹)': [cost['baseline_total_cost'], cost['ml_total_cost']],
    })
    st.subheader(f"Cost Breakdown at the Optimal {best_margin:.0%} Margin")
    st.dataframe(breakdown.style.format({c: '₹{:,.0f}' for c in breakdown.columns[1:]}),
                 use_container_width=True)


def render_model_performance(model_info):
    st.header("⚙️ Model Performance & Explainability")
    if model_info is None:
        st.warning("Model not found.")
        return

    st.write(f"**Selected Model:** {model_info['model_name']}")

    metrics = model_info.get('metrics')
    if metrics:
        st.subheader("Held-Out Test-Set Metrics")
        metrics_df = pd.DataFrame(metrics).T[['mae', 'rmse', 'mape', 'r2']]
        metrics_df.index = ['Naive Baseline', 'Random Forest', 'Gradient Boosting']
        metrics_df.columns = ['MAE', 'RMSE', 'MAPE (%)', 'R²']
        st.dataframe(metrics_df.style.format('{:.2f}'), use_container_width=True)
        st.caption("Naive baseline = 'same as last week'. Both ML models are tuned with "
                   "RandomizedSearchCV using time-series cross-validation.")

    best_params = model_info.get('best_params')
    if best_params:
        with st.expander("Selected hyperparameters"):
            st.json(best_params)

    importance = model_info.get('feature_importance')
    if importance:
        st.subheader("Feature Importance (permutation, test set)")
        top = importance[:15][::-1]
        fig = px.bar(x=[s for _, s in top], y=[n for n, _ in top], orientation='h',
                     title="Top Drivers of Demand",
                     labels={'x': 'Importance (error increase when shuffled)', 'y': 'Feature'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance not available. Re-run `python src/train.py`.")


def main():
    st.sidebar.title("🍔 Restaurant Demand")

    # Dataset selector — only offer sources whose files exist.
    available = {s: c['label'] for s, c in DATASETS.items() if os.path.exists(resolve(c['sales']))}
    if not available:
        st.error("No data found. Run `python src/data_generation.py` (and optionally "
                 "`python src/prepare_real_data.py`), then `python src/train.py`.")
        return
    source = st.sidebar.selectbox("Dataset", list(available), format_func=lambda s: available[s])

    st.sidebar.divider()
    page = st.sidebar.radio("Go to", ["Overview", "Predictions", "Waste Insights", "Cost Impact", "Model Performance"])

    sales_df, menu_df = load_sales_menu(source)
    st.sidebar.caption(f"{sales_df['item_id'].nunique()} items · "
                       f"{sales_df['date'].min().date()} → {sales_df['date'].max().date()}")

    if page == "Overview":
        render_overview(sales_df, menu_df)
        return

    model_info = load_model_bundle(source)
    test_data = score_test_data(source)
    if page == "Predictions":
        render_predictions(test_data, menu_df)
    elif page == "Waste Insights":
        render_waste_insights(test_data, menu_df)
    elif page == "Cost Impact":
        render_cost_impact(test_data, menu_df)
    elif page == "Model Performance":
        render_model_performance(model_info)


if __name__ == "__main__":
    main()
