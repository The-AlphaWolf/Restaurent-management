import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Add src to path to import modules
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from features import preprocess_data
from predict import predict_demand
from waste_optimizer import evaluate_waste_reduction

# Page config
st.set_page_config(page_title="Smart Restaurant Food Demand Prediction", layout="wide", page_icon="🍔")

# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

@st.cache_data
def load_data():
    sales_path = os.path.join(DATA_DIR, 'sales_data.csv')
    menu_path = os.path.join(DATA_DIR, 'menu_items.csv')
    if not os.path.exists(sales_path):
        return None, None
    sales_df = pd.read_csv(sales_path)
    sales_df['date'] = pd.to_datetime(sales_df['date'])
    menu_df = pd.read_csv(menu_path)
    return sales_df, menu_df

@st.cache_resource
def load_ml_model():
    model_path = os.path.join(MODEL_DIR, 'best_model.pkl')
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

@st.cache_data
def load_processed_data():
    sales_path = os.path.join(DATA_DIR, 'sales_data.csv')
    if not os.path.exists(sales_path):
        return None
    return preprocess_data(sales_path)

def render_overview(sales_df, menu_df):
    st.header("📈 Historical Sales Overview")
    
    # Merge for names
    df = sales_df.merge(menu_df[['item_id', 'name']], on='item_id')
    
    col1, col2 = st.columns(2)
    with col1:
        # Total Sales over time
        daily_total = df.groupby('date')['units_sold'].sum().reset_index()
        fig = px.line(daily_total, x='date', y='units_sold', title="Total Daily Demand Over Time")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Category breakdown
        category_total = df.groupby('category')['units_sold'].sum().reset_index()
        fig2 = px.pie(category_total, values='units_sold', names='category', title="Sales by Category")
        st.plotly_chart(fig2, use_container_width=True)
        
    st.subheader("Filter by Item")
    selected_item = st.selectbox("Select an item", df['name'].unique())
    item_data = df[df['name'] == selected_item]
    fig3 = px.line(item_data, x='date', y='units_sold', title=f"Daily Demand for {selected_item}")
    st.plotly_chart(fig3, use_container_width=True)

def render_predictions(processed_df, menu_df, model_info):
    st.header("🔮 Demand Predictions")
    st.write("Using the trained ML model to forecast demand based on historical patterns, seasonality, and weather.")
    
    if processed_df is None or model_info is None:
        st.warning("Data or model not found. Please run the training pipeline.")
        return
        
    # We will simulate predictions for the test set (last 6 months)
    test_data = processed_df[processed_df['date'] >= '2023-07-01'].copy()
    
    if len(test_data) == 0:
        st.warning("No test data available after 2023-07-01.")
        return
        
    test_data['predicted_demand'] = predict_demand(test_data, model_info)
    
    # Merge with menu items for readability
    test_data = test_data.merge(menu_df[['item_id', 'name', 'category']], on='item_id')
    
    selected_category = st.selectbox("Select Category", ['All'] + list(test_data['category'].unique()))
    
    if selected_category != 'All':
        display_data = test_data[test_data['category'] == selected_category]
    else:
        display_data = test_data
        
    # Aggregate daily totals for the view
    daily_actual = display_data.groupby('date')['units_sold'].sum().reset_index()
    daily_pred = display_data.groupby('date')['predicted_demand'].sum().reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily_actual['date'], y=daily_actual['units_sold'], mode='lines', name='Actual Demand'))
    fig.add_trace(go.Scatter(x=daily_pred['date'], y=daily_pred['predicted_demand'], mode='lines', name='Predicted Demand', line=dict(dash='dash')))
    fig.update_layout(title="Actual vs Predicted Demand", xaxis_title="Date", yaxis_title="Units Sold")
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Item View
    st.subheader("Item-Level Forecasts")
    selected_item = st.selectbox("Select an item to view specific predictions", display_data['name'].unique(), key='item_pred')
    item_pred_data = display_data[display_data['name'] == selected_item].tail(14) # Show last 14 days
    
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

def render_waste_insights(processed_df, menu_df, model_info):
    st.header("♻️ Waste Reduction Optimizer")
    
    if processed_df is None or model_info is None:
        st.warning("Data or model not found.")
        return
        
    # Apply to test set
    test_data = processed_df[processed_df['date'] >= '2023-07-01'].copy()
    test_data['predicted_demand'] = predict_demand(test_data, model_info)
    
    safety_margin = st.slider("Safety Margin (%)", min_value=0, max_value=50, value=10, step=5) / 100.0
    
    eval_df, metrics = evaluate_waste_reduction(test_data, safety_margin=safety_margin)
    
    st.subheader("Impact of ML-Driven Preparation")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Baseline Strategy Waste", f"{metrics['total_baseline_waste_units']:,.0f} units")
    col2.metric("ML Strategy Waste", f"{metrics['total_ml_waste_units']:,.0f} units", f"-{metrics['total_baseline_waste_units'] - metrics['total_ml_waste_units']:,.0f} units", delta_color="inverse")
    col3.metric("Waste Reduction", f"{metrics['waste_reduction_percent']:.1f}%")
    
    st.info("The Baseline Strategy assumes preparing the maximum daily quantity sold over the prior 14 days. The ML Strategy prepares the predicted demand plus the safety margin.")
    
    # Merge for names
    eval_df = eval_df.merge(menu_df[['item_id', 'name']], on='item_id')
    
    st.subheader("Top Waste Reduction Opportunities")
    # Group by item
    item_waste = eval_df.groupby('name')[['baseline_waste', 'ml_waste']].sum().reset_index()
    item_waste['reduction'] = item_waste['baseline_waste'] - item_waste['ml_waste']
    item_waste = item_waste.sort_values(by='reduction', ascending=False).head(10)
    
    fig = px.bar(item_waste, x='reduction', y='name', orientation='h', title="Top 10 Items for Waste Reduction",
                 labels={'reduction': 'Units Saved', 'name': 'Item'})
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

def render_model_performance(model_info):
    st.header("⚙️ Model Performance & Explainability")
    
    if model_info is None:
        st.warning("Model not found.")
        return
        
    st.write(f"**Current Model:** {model_info['model_name']}")
    
    model = model_info['model']
    features = model_info['features']
    
    if hasattr(model, 'feature_importances_'):
        st.subheader("Feature Importance")
        importances = model.feature_importances_
        # Sort features
        indices = np.argsort(importances)[::-1]
        
        # Take top 15
        top_indices = indices[:15]
        top_features = [features[i] for i in top_indices]
        top_importances = importances[top_indices]
        
        fig = px.bar(x=top_importances, y=top_features, orientation='h', 
                     title="Top Drivers of Food Demand",
                     labels={'x': 'Importance Score', 'y': 'Feature'})
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Feature importance not available for this model type.")

def main():
    sales_df, menu_df = load_data()
    
    if sales_df is None:
        st.error("Data files not found. Please run `python src/data_generation.py` to generate the synthetic dataset.")
        return
        
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Overview", "Predictions", "Waste Insights", "Model Performance"])
    
    if page == "Overview":
        render_overview(sales_df, menu_df)
    else:
        model_info = load_ml_model()
        processed_df = load_processed_data()
        
        if page == "Predictions":
            render_predictions(processed_df, menu_df, model_info)
        elif page == "Waste Insights":
            render_waste_insights(processed_df, menu_df, model_info)
        elif page == "Model Performance":
            render_model_performance(model_info)

if __name__ == "__main__":
    main()
