"""
Waste Optimization module.

Calculates recommended prep quantities, estimates potential food waste,
and compares the ML-driven approach against naive baselines to quantify
the value of the prediction system.
"""

import pandas as pd
import numpy as np

def calculate_prep_quantity(predictions, safety_margin=0.10):
    """
    Calculates the recommended preparation quantity.
    Prep = Predicted Demand * (1 + safety_margin)
    """
    return np.ceil(np.array(predictions) * (1.0 + safety_margin))

def simulate_waste(actual_demand, prep_quantity):
    """
    Simulates waste and stockouts.
    Waste occurs if we prep more than we sell.
    """
    actual = np.array(actual_demand)
    prep = np.array(prep_quantity)
    
    waste = np.maximum(0, prep - actual)
    stockout = np.maximum(0, actual - prep)
    
    return waste, stockout

def evaluate_waste_reduction(df, predicted_col='predicted_demand', actual_col='units_sold', safety_margin=0.10):
    """
    Compares the ML strategy against a baseline strategy.
    Baseline Strategy: Prepare the maximum amount sold in the last 14 days (rolling_max).
    """
    df = df.copy()
    
    # Ensure data is sorted
    df = df.sort_values(by=['item_id', 'date']).reset_index(drop=True)
    
    # Calculate baseline prep (rolling 14-day max)
    # Shift by 1 so we don't include today's actual demand
    df['baseline_prep'] = df.groupby('item_id')[actual_col].transform(
        lambda x: x.shift(1).rolling(window=14, min_periods=1).max()
    )
    # Fill any NaNs with the actual demand just to have a baseline
    df['baseline_prep'] = df['baseline_prep'].fillna(df[actual_col])
    
    # Calculate ML prep
    df['ml_prep'] = calculate_prep_quantity(df[predicted_col], safety_margin=safety_margin)
    
    # Simulate waste for both strategies
    df['ml_waste'], df['ml_stockout'] = simulate_waste(df[actual_col], df['ml_prep'])
    df['baseline_waste'], df['baseline_stockout'] = simulate_waste(df[actual_col], df['baseline_prep'])
    
    # Aggregate metrics
    total_ml_waste = df['ml_waste'].sum()
    total_baseline_waste = df['baseline_waste'].sum()
    
    total_ml_stockout = df['ml_stockout'].sum()
    total_baseline_stockout = df['baseline_stockout'].sum()
    
    waste_reduction_percent = 0
    if total_baseline_waste > 0:
        waste_reduction_percent = ((total_baseline_waste - total_ml_waste) / total_baseline_waste) * 100
        
    metrics = {
        'total_ml_waste_units': total_ml_waste,
        'total_baseline_waste_units': total_baseline_waste,
        'waste_reduction_percent': waste_reduction_percent,
        'total_ml_stockout_units': total_ml_stockout,
        'total_baseline_stockout_units': total_baseline_stockout,
    }
    
    return df, metrics
