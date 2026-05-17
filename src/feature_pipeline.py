
"""
Feature Pipeline: Fetch + Clean + Feature Engineer + Store to MongoDB
Location: Islamabad, Pakistan
Runs: Every 5 hours via GitHub Actions (incremental)
      Once manually for historical backfill (Jan 2026 - May 2026)
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pymongo import MongoClient
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

load_dotenv()  # ← must come after imports
# ============================================
# CONFIGURATION
# ============================================
LAT      = 33.7294
LON      = 73.0931
TIMEZONE = "Asia/Karachi"

# Feature columns for model
FEATURE_COLS = [
    'lag1', 'lag2', 'lag3', 'lag6', 'lag12', 'lag24', 'lag48', 'lag72',
    'aqi_ma6', 'aqi_ma12', 'aqi_ma24', 'aqi_std12',
    'hour', 'day_of_week', 'month',
    'pm2_5', 'pm10', 'o3', 'temperature', 'wind_speed', 'rain_code'
]

# ============================================
# MONGODB CONNECTION
# ============================================
def get_mongo_collection():
    """Connect to MongoDB and return the feature collection"""

    mongo_uri = os.environ.get("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("MONGODB_URI environment variable not set!")

    client     = MongoClient(mongo_uri)
    db         = client["aqi_db"]
    collection = db["aqi_features"]

    # Create index on datetime to avoid duplicates and speed up queries
    collection.create_index("datetime", unique=True)

    print("   Connected to MongoDB")
    return collection

# ============================================
# STEP 1: FETCH RAW DATA
# ============================================
def fetch_raw_data(start_date, end_date):
    """Fetch hourly air quality + weather data from Open-Meteo"""

    print(f"  Fetching data from {start_date} to {end_date}...")

    # 1. Air Quality
    print("  Fetching air quality...")
    aq_resp = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            "latitude":  LAT,
            "longitude": LON,
            "hourly":    "pm2_5,pm10,nitrogen_dioxide,ozone,carbon_monoxide,us_aqi",
            "start_date": start_date,
            "end_date":   end_date,
            "timezone":   TIMEZONE,
        },
        timeout=30,
    )
    aq_resp.raise_for_status()
    aq = aq_resp.json()["hourly"]

    # 2. Weather
    print("  Fetching weather...")
    wx_resp = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude":  LAT,
            "longitude": LON,
            "hourly":    "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
            "start_date": start_date,
            "end_date":   end_date,
            "timezone":   TIMEZONE,
        },
        timeout=30,
    )
    wx_resp.raise_for_status()
    wx = wx_resp.json()["hourly"]

    # 3. Build DataFrame
    df = pd.DataFrame({
        "datetime":      pd.to_datetime(aq["time"]),
        "aqi":           aq["us_aqi"],
        "pm2_5":         aq["pm2_5"],
        "pm10":          aq["pm10"],
        "no2":           aq["nitrogen_dioxide"],
        "o3":            aq["ozone"],
        "co":            aq["carbon_monoxide"],
        "temperature":   wx["temperature_2m"],
        "humidity":      wx["relative_humidity_2m"],
        "wind_speed":    wx["wind_speed_10m"],
        "precipitation": wx["precipitation"],
    })

    print(f"   Fetched {len(df)} rows")
    return df

# ============================================
# STEP 2: CLEANING
# ============================================
def clean_data(df):
    """Handle missing values and duplicates"""

    # Remove duplicates
    df = df.drop_duplicates(subset=['datetime'])

    # Fill missing values for key columns with median
    for col in ['aqi', 'pm2_5', 'pm10', 'temperature', 'wind_speed']:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Forward fill then backward fill remaining nulls
    df = df.ffill().bfill()

    print(f"   After cleaning: {len(df)} rows")
    return df

# ============================================
# STEP 3: FEATURE ENGINEERING
# ============================================
def create_features(df):
    """Create all features needed for model"""

    df = df.sort_values('datetime').reset_index(drop=True)

    # Time features
    df['hour']        = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month']       = df['datetime'].dt.month
    df['day']         = df['datetime'].dt.day

    # Rain code (categorize precipitation)
    def get_rain_code(precip):
        if precip == 0:      return 0  # dry
        elif precip < 2.5:   return 1  # light rain
        elif precip < 10:    return 2  # moderate rain
        else:                return 3  # heavy rain

    df['rain_code'] = df['precipitation'].apply(get_rain_code)

    # Lag features (past AQI values)
    for lag in [1, 2, 3, 6, 12, 24, 48, 72]:
        df[f'lag{lag}'] = df['aqi'].shift(lag)

    # Rolling statistics
    df['aqi_ma6']   = df['aqi'].rolling(6).mean()
    df['aqi_ma12']  = df['aqi'].rolling(12).mean()
    df['aqi_ma24']  = df['aqi'].rolling(24).mean()
    df['aqi_std12'] = df['aqi'].rolling(12).std()

    # Target variables (future AQI to predict)
    df['target_h24'] = df['aqi'].shift(-24)
    df['target_h48'] = df['aqi'].shift(-48)
    df['target_h72'] = df['aqi'].shift(-72)

    # Drop rows with NaN (from lag/rolling/target creation)
    df = df.dropna().reset_index(drop=True)

    print(f"   After feature engineering: {len(df)} rows")
    print(f"   Total columns: {len(df.columns)}")
    return df

# ============================================
# STEP 4: STORE TO MONGODB
# ============================================
def store_to_mongodb(df, collection):
    """Store processed features to MongoDB — skips duplicates"""

    # Select columns to store
    store_cols = ['datetime'] + FEATURE_COLS + ['target_h24', 'target_h48', 'target_h72']
    df_store   = df[store_cols].copy()

    # Convert datetime to string for MongoDB storage
    df_store['datetime'] = df_store['datetime'].astype(str)

    # Convert to list of dicts
    records = df_store.to_dict('records')

    # Insert — skip duplicates using unique index on datetime
    inserted = 0
    skipped  = 0

    for record in records:
        try:
            collection.insert_one(record)
            inserted += 1
        except Exception:
            skipped += 1  # duplicate datetime — skip silently

    print(f"   Inserted: {inserted} new rows")
    print(f"    Skipped: {skipped} duplicate rows")
    return inserted

# ============================================
# STEP 5: MAIN PIPELINE
# ============================================
def run_feature_pipeline(start_date=None, end_date=None):
    """
    Main pipeline execution.

    Historical backfill (run once manually):
        run_feature_pipeline("2026-01-01", "2026-05-16")

    Incremental (runs every 5 hours via GitHub Actions):
        run_feature_pipeline()
    """

    print("\n" + "="*60)
    print("FEATURE PIPELINE — ISLAMABAD AQI")
    print("="*60)

    # Set dates
    if start_date is None:
        # Incremental: fetch last 8 hours (overlap buffer for safety)
        end_dt    = datetime.now()
        start_dt  = end_dt - timedelta(hours=8)
        start_date = start_dt.strftime("%Y-%m-%d")
        end_date   = end_dt.strftime("%Y-%m-%d")
        print(f"Mode      : INCREMENTAL (last 8 hours with buffer)")
    else:
        print(f"Mode      : HISTORICAL BACKFILL")

    print(f"Date range: {start_date} → {end_date}")
    print("-"*60)

    # Connect to MongoDB
    print("\n[1/4] Connecting to MongoDB...")
    collection = get_mongo_collection()

    # Step 1: Fetch raw data
    print("\n[2/4] Fetching raw data...")
    df_raw = fetch_raw_data(start_date, end_date)

    # Step 2: Clean
    print("\n[3/4] Cleaning data...")
    df_clean = clean_data(df_raw)

    # Step 3: Feature engineering
    print("\n[4/4] Engineering features...")
    df_features = create_features(df_clean)

    # Step 4: Store to MongoDB
    print("\n[5/4] Storing to MongoDB...")
    store_to_mongodb(df_features, collection)

    # Summary
    print("\n" + "="*60)
    print(" FEATURE PIPELINE COMPLETED SUCCESSFULLY")
    print(f"   Rows processed : {len(df_features)}")
    print(f"   Columns        : {len(df_features.columns)}")
    print(f"   Time           : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    return df_features


# ============================================
# RUN
# ============================================
if __name__ == "__main__":

    # ▶ HISTORICAL BACKFILL — run this first time only
    #run_feature_pipeline("2026-01-01", "2026-05-16")

    # ▶ INCREMENTAL — GitHub Actions runs this every 5 hours
     run_feature_pipeline()