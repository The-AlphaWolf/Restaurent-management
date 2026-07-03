"""
Prediction module.

Loads the trained model and generates predictions for given data.
"""

import os
import joblib
import pandas as pd

def load_model(model_path=None):
    if model_path is None:
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'best_model.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please run train.py first.")
        
    return joblib.load(model_path)

def predict_demand(df, model_info=None):
    """
    Generates predictions for a preprocessed dataframe.
    """
    if model_info is None:
        model_info = load_model()
        
    model = model_info['model']
    features = model_info['features']
    
    # Ensure all required features are present
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features in dataframe: {missing_features}")
        
    X = df[features]
    predictions = model.predict(X)
    
    # Demand cannot be negative
    predictions = [max(0, p) for p in predictions]
    
    return predictions
