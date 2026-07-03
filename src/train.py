"""
Model Training script.

Trains machine learning models to predict per-item daily demand.
Compares Random Forest and LightGBM, evaluates on a chronological test set,
and saves the best model.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from features import preprocess_data
from datetime import timedelta

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    non_zero_idx = y_true != 0
    if sum(non_zero_idx) == 0:
        return 0.0
    return np.mean(np.abs((y_true[non_zero_idx] - y_pred[non_zero_idx]) / y_true[non_zero_idx])) * 100

def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"--- {name} ---")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R2:   {r2:.4f}")
    print("-" * 20)
    
    return {'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2}

def get_train_test_split(df, split_date='2023-07-01'):
    """Splits data chronologically."""
    train = df[df['date'] < split_date].copy()
    test = df[df['date'] >= split_date].copy()
    return train, test

def main():
    print("Loading and preprocessing data...")
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'sales_data.csv')
    df = preprocess_data(data_path)
    
    # Chronological split (last 6 months for testing)
    train_df, test_df = get_train_test_split(df, split_date='2023-07-01')
    
    # Define features and target
    # Exclude non-feature columns
    features_to_drop = ['date', 'item_id', 'units_sold']
    features = [col for col in df.columns if col not in features_to_drop]
    
    X_train = train_df[features]
    y_train = train_df['units_sold']
    
    X_test = test_df[features]
    y_test = test_df['units_sold']
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Baseline: Naive forecast (predict same as 7 days ago)
    print("\nEvaluating Baseline (Same as last week)...")
    baseline_preds = test_df['lag_7']
    evaluate_model("Baseline (lag_7)", y_test, baseline_preds)
    
    # Train Random Forest
    print("\nTraining Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_metrics = evaluate_model("Random Forest", y_test, rf_preds)
    
    # Train LightGBM
    print("\nTraining LightGBM...")
    # Using specific params that usually work well for this kind of tabular data
    lgb_model = LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=8, random_state=42)
    lgb_model.fit(X_train, y_train)
    lgb_preds = lgb_model.predict(X_test)
    lgb_metrics = evaluate_model("LightGBM", y_test, lgb_preds)
    
    # Select best model (based on MAE)
    if lgb_metrics['mae'] <= rf_metrics['mae']:
        best_model = lgb_model
        best_name = "LightGBM"
    else:
        best_model = rf_model
        best_name = "Random Forest"
        
    print(f"\nBest model selected: {best_name}")
    
    # Save the model
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_info = {
        'model': best_model,
        'features': features,
        'model_name': best_name
    }
    
    model_path = os.path.join(models_dir, 'best_model.pkl')
    joblib.dump(model_info, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
