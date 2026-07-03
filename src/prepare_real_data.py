"""
Prepare a real restaurant-demand dataset.

Source: the **Recruit Restaurant Visitor Forecasting** data (AirREGI point-of-sale
system, Japan) — real daily visitor counts per restaurant, plus a holiday
calendar and per-store cuisine genre. This is a genuine demand time series at
the same daily granularity as our synthetic data, so it flows through the exact
same feature/model pipeline.

What this script does:
1. Downloads the three source CSVs (cached locally after the first run).
2. Maps them onto the project's canonical schema.
3. Subsamples to the busiest, most-complete restaurants to keep the repo small.
4. Assigns an INR average-check per cuisine genre (a *documented assumption*,
   since visitor data has no prices) so the cost module works on real data too.

Outputs:
- ``data/real_sales_data.csv``  (date, item_id, category, units_sold, is_holiday)
- ``data/real_menu_items.csv``  (item_id, category, name, selling_price, food_cost)
"""

from __future__ import annotations

import os
import urllib.request

import pandas as pd

from datasets import resolve

BASE_URL = "https://raw.githubusercontent.com/missingfactor/raw-data/master"
SOURCE_FILES = ["air_visit_data.csv", "date_info.csv", "air_store_info.csv"]
CACHE_DIR = resolve("data/raw_recruit")

# How many restaurants to keep, and the minimum days of history each must have.
N_STORES = 30
MIN_DAYS = 300

# Average spend per visitor (INR) by cuisine genre — a modelling assumption used
# only to demonstrate the cost module on price-less visitor data.
GENRE_PRICE_INR = {
    "Izakaya": 900,
    "Cafe/Sweets": 400,
    "Dining bar": 1100,
    "Italian/French": 1300,
    "Bar/Cocktail": 1000,
    "Japanese food": 850,
    "Yakiniku/Korean food": 1200,
    "Western food": 950,
    "Creative cuisine": 1400,
    "Okonomiyaki/Monja/Teppanyaki": 800,
    "International cuisine": 1250,
    "Asian": 750,
    "Karaoke/Party": 700,
}
DEFAULT_PRICE_INR = 900
FOOD_COST_RATIO = 0.35  # visitor-level COGS assumption


def _download(filename: str) -> str:
    """Download ``filename`` into the cache dir if not already present."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    dest = os.path.join(CACHE_DIR, filename)
    if not os.path.exists(dest):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(f"{BASE_URL}/{filename}", dest)
    return dest


def load_source_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Download (if needed) and load the three raw Recruit CSVs."""
    visits = pd.read_csv(_download("air_visit_data.csv"))
    dates = pd.read_csv(_download("date_info.csv"))
    stores = pd.read_csv(_download("air_store_info.csv"))
    return visits, dates, stores


def build_canonical(
    visits: pd.DataFrame, dates: pd.DataFrame, stores: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map the raw frames to the canonical sales + menu schema."""
    visits = visits.rename(columns={"visit_date": "date", "visitors": "units_sold"})
    visits["date"] = pd.to_datetime(visits["date"])

    # Keep the busiest, most-complete restaurants for a clean, compact demo.
    counts = visits.groupby("air_store_id")["date"].count()
    eligible = counts[counts >= MIN_DAYS].sort_values(ascending=False).head(N_STORES).index
    visits = visits[visits["air_store_id"].isin(eligible)].copy()

    # Attach the holiday flag from the date calendar.
    dates = dates.rename(columns={"calendar_date": "date", "holiday_flg": "is_holiday"})
    dates["date"] = pd.to_datetime(dates["date"])
    visits = visits.merge(dates[["date", "is_holiday"]], on="date", how="left")

    # Attach cuisine genre (our "category") and build a readable item id/name.
    stores = stores.rename(columns={"air_genre_name": "category"})
    visits = visits.merge(stores[["air_store_id", "category", "air_area_name"]], on="air_store_id")

    # Stable, readable item ids (STORE_01 ...) mapped from the store hashes.
    id_map = {sid: f"STORE_{i + 1:02d}" for i, sid in enumerate(sorted(eligible))}
    visits["item_id"] = visits["air_store_id"].map(id_map)

    sales = visits[["date", "item_id", "category", "units_sold", "is_holiday"]].copy()
    sales["is_holiday"] = sales["is_holiday"].fillna(0).astype(int)
    sales = sales.sort_values(["date", "item_id"]).reset_index(drop=True)

    # Build the menu table with INR price assumptions.
    menu = visits[["item_id", "category", "air_area_name"]].drop_duplicates("item_id").copy()
    menu["name"] = menu["item_id"] + " · " + menu["category"]
    menu["selling_price"] = menu["category"].map(GENRE_PRICE_INR).fillna(DEFAULT_PRICE_INR)
    menu["food_cost"] = (menu["selling_price"] * FOOD_COST_RATIO).round(2)
    menu = menu[["item_id", "category", "name", "selling_price", "food_cost"]].reset_index(drop=True)

    return sales, menu


def main() -> None:
    visits, dates, stores = load_source_frames()
    sales, menu = build_canonical(visits, dates, stores)

    sales_path = resolve("data/real_sales_data.csv")
    menu_path = resolve("data/real_menu_items.csv")
    sales.to_csv(sales_path, index=False)
    menu.to_csv(menu_path, index=False)

    print(f"Saved {sales_path} ({len(sales):,} rows, {sales['item_id'].nunique()} restaurants)")
    print(f"Saved {menu_path}")
    print(f"Date range: {sales['date'].min().date()} -> {sales['date'].max().date()}")


if __name__ == "__main__":
    main()
