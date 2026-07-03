"""
Feature Engineering module.

Adds lag features, rolling statistics, and prepares the dataset for machine learning models.
"""

import pandas as pd
import numpy as np

def create_features(df):
    """
    Takes the raw sales dataframe and creates time-series features.
    Features created:
    - Lags: demand 1 day ago, 7 days ago, 14 days ago.
    - Rolling averages: 7-day and 14-day rolling mean of demand.
    """
    # Ensure dataframe is sorted chronologically per item
    df = df.sort_values(by=['item_id', 'date']).reset_index(drop=True)
    
    # Create lag features (shift by 1, 7, 14 days)
    df['lag_1'] = df.groupby('item_id')['units_sold'].shift(1)
    df['lag_7'] = df.groupby('item_id')['units_sold'].shift(7)
    df['lag_14'] = df.groupby('item_id')['units_sold'].shift(14)
    
    # Create rolling features (shift by 1 first so we don't include the target day in the rolling mean)
    # The rolling mean of the past 7 days, excluding today.
    df['rolling_mean_7'] = df.groupby('item_id')['units_sold'].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()
    )
    df['rolling_mean_14'] = df.groupby('item_id')['units_sold'].transform(
        lambda x: x.shift(1).rolling(window=14, min_periods=1).mean()
    )
    
    # Fill NA values that resulted from shifting and rolling
    # We can fill them with 0, or drop them. Let's drop rows with NaNs as we have enough data (3 years)
    # But wait, dropping will drop the first 14 days of the entire dataset. That's acceptable.
    df = df.dropna().reset_index(drop=True)
    
    return df

def preprocess_data(filepath='data/sales_data.csv'):
    """
    Loads data, applies feature engineering, and encodes categorical variables.
    """
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    
    # Generate features
    df = create_features(df)
    
    # One-hot encode the category variable if we want to use it as a feature
    # Actually, for tree-based models like Random Forest, label encoding or one-hot works.
    # We will use one-hot for simplicity in scikit-learn.
    df = pd.get_dummies(df, columns=['category'], drop_first=True)
    
    # Drop columns that are not features (like the target, item_id string, date)
    # We will keep 'item_id' for grouping and separating models if needed, 
    # but the instructions said "one model with item as feature". 
    # Let's label encode 'item_id' to use it as a feature.
    
    # Convert item_id to numeric category
    df['item_id_encoded'] = df['item_id'].astype('category').cat.codes
    
    return df

if __name__ == "__main__":
    import os
    # Quick test to see if it works
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'sales_data.csv')
    if os.path.exists(data_path):
        processed_df = preprocess_data(data_path)
        print(f"Processed dataframe shape: {processed_df.shape}")
        print(processed_df.head())
    else:
        print(f"File not found: {data_path}. Run data_generation.py first.")
