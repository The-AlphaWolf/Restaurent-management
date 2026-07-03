import os
import sys
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from data_generation import generate_menu_items, generate_weather
from features import create_features
from waste_optimizer import calculate_prep_quantity, simulate_waste

def test_generate_menu_items():
    df = generate_menu_items()
    assert not df.empty
    assert 'item_id' in df.columns
    assert 'price' in df.columns
    assert len(df) >= 15 # We have around 20 items

def test_generate_weather():
    dates = pd.date_range('2023-01-01', '2023-12-31')
    temp = generate_weather(dates)
    assert len(temp) == 365
    # Check if temps are in a reasonable synthetic range
    assert temp.min() > -20
    assert temp.max() < 120

def test_create_features():
    # Create dummy data
    data = {
        'date': pd.date_range('2023-01-01', periods=20),
        'item_id': ['ITEM_001'] * 20,
        'units_sold': np.random.randint(10, 50, 20)
    }
    df = pd.DataFrame(data)
    
    features_df = create_features(df)
    
    # Check if features were created
    assert 'lag_1' in features_df.columns
    assert 'rolling_mean_7' in features_df.columns
    
    # 14 days should be dropped due to na in rolling 14 / lag 14
    assert len(features_df) == 6

def test_calculate_prep_quantity():
    predictions = [100, 50, 0]
    prep = calculate_prep_quantity(predictions, safety_margin=0.1)
    
    assert list(prep) == [111.0, 55.0, 0.0]

def test_simulate_waste():
    actual = [100, 60, 40]
    prep = [110, 50, 40]
    
    waste, stockout = simulate_waste(actual, prep)
    
    assert list(waste) == [10.0, 0.0, 0.0]
    assert list(stockout) == [0.0, 10.0, 0.0]
