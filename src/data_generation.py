"""
Synthetic Data Generation for Smart Restaurant Food Demand Prediction System.

This script generates ~3 years of realistic daily sales data for a hypothetical restaurant.
It creates a synthetic dataset designed to mimic real-world restaurant demand patterns:
- Weekly seasonality (busier on weekends).
- Yearly seasonality (shifts in demand based on seasons and specific holidays).
- Overall growth trend (restaurant popularity increasing over time).
- Weather influence (synthetic temperature affecting certain categories, e.g., cold beverages sell more when hot).
- Random noise to simulate real-world variability.

Outputs two CSV files in the `data/` directory:
- `menu_items.csv`: Details of the menu items (category, base price).
- `sales_data.csv`: Daily sales records per item.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# --- Configuration ---
START_DATE = datetime(2021, 1, 1)
END_DATE = datetime(2023, 12, 31)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

MENU_CONFIG = {
    'Mains': [
        ('Grilled Salmon', 25.0, 1.2, 'stable'),
        ('Beef Burger', 15.0, 1.5, 'stable'),
        ('Mushroom Risotto', 18.0, 1.0, 'winter_high'),
        ('Chicken Caesar Salad', 14.0, 1.3, 'summer_high'),
        ('Steak Frites', 28.0, 0.9, 'stable'),
        ('Vegetarian Moussaka', 16.0, 0.8, 'stable')
    ],
    'Appetizers': [
        ('Calamari', 12.0, 1.4, 'stable'),
        ('Bruschetta', 9.0, 1.2, 'summer_high'),
        ('French Onion Soup', 8.0, 1.5, 'winter_high'),
        ('Garlic Bread', 6.0, 2.0, 'stable')
    ],
    'Desserts': [
        ('Cheesecake', 8.0, 1.0, 'stable'),
        ('Chocolate Fondant', 9.0, 1.1, 'winter_high'),
        ('Ice Cream Sundae', 7.0, 1.3, 'summer_high'),
        ('Tiramisu', 8.5, 0.9, 'stable')
    ],
    'Beverages': [
        ('Craft Beer', 7.0, 2.5, 'summer_high'),
        ('House Red Wine', 8.0, 2.0, 'winter_high'),
        ('House White Wine', 8.0, 1.8, 'summer_high'),
        ('Lemonade', 4.0, 1.5, 'summer_high'),
        ('Hot Coffee', 3.5, 2.0, 'winter_high')
    ]
}

HOLIDAYS = {
    'New Year': (1, 1),
    'Valentines': (2, 14),
    'Independence Day': (7, 4),
    'Halloween': (10, 31),
    'Thanksgiving': (11, 25), # Simplified to specific date for synthetic data
    'Christmas': (12, 25),
    'New Year Eve': (12, 31)
}


def generate_menu_items():
    """Generates the menu items dataframe."""
    items = []
    item_id = 1
    for category, food_list in MENU_CONFIG.items():
        for name, price, base_popularity, seasonality_type in food_list:
            items.append({
                'item_id': f'ITEM_{item_id:03d}',
                'category': category,
                'name': name,
                'price': price,
                'base_popularity': base_popularity,
                'seasonality_type': seasonality_type
            })
            item_id += 1
    return pd.DataFrame(items)


def generate_weather(dates):
    """Generates synthetic temperature data based on a yearly sine wave."""
    # Assuming Northern Hemisphere: peak temp in July (day 200), lowest in Jan (day 15)
    days_of_year = dates.dayofyear
    # Sine wave scaled to simulate temperature roughly between 30F and 85F
    temp = 55 + 30 * np.sin(2 * np.pi * (days_of_year - 105) / 365)
    # Add random daily variation
    temp += np.random.normal(0, 5, len(dates))
    return temp


def generate_sales_data(menu_df, start_date, end_date):
    """Generates daily sales data per item."""
    dates = pd.date_range(start_date, end_date)
    
    # 1. Generate general environmental features
    env_df = pd.DataFrame({'date': dates})
    env_df['temperature'] = generate_weather(dates)
    env_df['day_of_week'] = env_df['date'].dt.dayofweek
    env_df['month'] = env_df['date'].dt.month
    env_df['is_weekend'] = env_df['day_of_week'].isin([4, 5, 6]).astype(int) # Fri, Sat, Sun are busy
    
    # Yearly trend: Linear growth factor starting at 1.0 and increasing by 10% each year
    total_days = len(dates)
    env_df['trend_factor'] = np.linspace(1.0, 1.3, total_days)
    
    # Holidays
    env_df['is_holiday'] = 0
    for holiday, (month, day) in HOLIDAYS.items():
        mask = (env_df['date'].dt.month == month) & (env_df['date'].dt.day == day)
        env_df.loc[mask, 'is_holiday'] = 1
    
    sales_records = []
    
    print(f"Generating data for {len(menu_df)} items over {total_days} days...")
    
    for _, item in menu_df.iterrows():
        item_id = item['item_id']
        category = item['category']
        base_pop = item['base_popularity']
        seasonality = item['seasonality_type']
        
        # Base demand scales with popularity
        daily_demand = np.full(total_days, 15.0 * base_pop)
        
        # Apply weekly seasonality (Weekends & Fridays are busier)
        weekly_multiplier = np.where(env_df['is_weekend'] == 1, 1.5, 0.8)
        daily_demand *= weekly_multiplier
        
        # Apply yearly seasonality based on item type
        if seasonality == 'summer_high':
            # Higher demand when temp is higher
            temp_factor = 1.0 + (env_df['temperature'] - 55) / 100 
            daily_demand *= temp_factor
        elif seasonality == 'winter_high':
            # Higher demand when temp is lower
            temp_factor = 1.0 + (55 - env_df['temperature']) / 100
            daily_demand *= temp_factor
            
        # Apply Holiday spikes
        # General bump on holidays
        daily_demand = np.where(env_df['is_holiday'] == 1, daily_demand * 1.3, daily_demand)
        
        # Apply overall growth trend
        daily_demand *= env_df['trend_factor']
        
        # Add random noise (Poisson-like distribution for count data)
        # We use a normal distribution for the mean, then draw from Poisson
        noise_factor = np.random.normal(1.0, 0.15, total_days)
        expected_demand = np.maximum(0, daily_demand * noise_factor)
        
        # Final units sold is a Poisson draw to ensure integers and realistic variance
        units_sold = np.random.poisson(expected_demand)
        
        # Create records for this item
        item_records = pd.DataFrame({
            'date': env_df['date'],
            'item_id': item_id,
            'category': category,
            'price': item['price'],
            'temperature': env_df['temperature'],
            'is_weekend': env_df['is_weekend'],
            'is_holiday': env_df['is_holiday'],
            'units_sold': units_sold
        })
        
        sales_records.append(item_records)
        
    final_df = pd.concat(sales_records, ignore_index=True)
    
    # Sort by date then item
    final_df = final_df.sort_values(by=['date', 'item_id']).reset_index(drop=True)
    return final_df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Generating menu items...")
    menu_df = generate_menu_items()
    menu_path = os.path.join(OUTPUT_DIR, 'menu_items.csv')
    # Save menu items without the base_popularity and seasonality internal features
    menu_df[['item_id', 'category', 'name', 'price']].to_csv(menu_path, index=False)
    print(f"Saved {menu_path}")
    
    print("Generating daily sales data...")
    sales_df = generate_sales_data(menu_df, START_DATE, END_DATE)
    sales_path = os.path.join(OUTPUT_DIR, 'sales_data.csv')
    sales_df.to_csv(sales_path, index=False)
    print(f"Saved {sales_path}")
    print(f"Total sales records: {len(sales_df)}")

if __name__ == "__main__":
    main()
