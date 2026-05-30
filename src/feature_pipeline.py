"""
Feature Pipeline: Fetch + Clean + Feature Engineer + Store to MongoDB
Location: Islamabad, Pakistan
Runs: Every 5 hours via GitHub Actions (incremental)
      Once manually for historical backfill
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

load_dotenv()

# ============================================
# CONFIGURATION
# ============================================
LAT      = 33.7294
LON      = 73.0931
TIMEZONE = "Asia/Karachi"

FEATURE_COLS = [
    'lag1', 'lag2', 'lag3', 'lag6', 'lag12', 'lag24', 'lag48', 'lag72',
    'aqi_ma6', 'aqi_ma12', 'aqi_ma24', 'aqi_std12',
    'aqi_trend_3h', 'aqi_trend_6h', 'aqi_trend_24h',
    'aqi_min_24h', 'aqi_max_24h', 'aqi_range_24h',
    'pm2_5', 'pm10',
    'pm25_lag24', 'pm25_ma12',
    'hour_sin', 'hour_cos',
    'season', 'is_rush_hour', 'is_smog_season',
    'day_of_week', 'o3', 'no2', 'co', 'temperature', 'wind_speed','humidity', 'rain_code'
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

    collection.create_index("datetime", unique=True)

    print("   Connected to MongoDB")
    return collection

# ============================================
# STEP 1: FETCH RAW DATA
# ============================================
def fetch_raw_data(start_date, end_date):
    """Fetch hourly air quality + weather data from Open-Meteo"""

    print(f"  Fetching data from {start_date} to {end_date}...")

    print("  Fetching air quality...")
    aq_resp = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            "latitude":   LAT,
            "longitude":  LON,
            "hourly":     "pm2_5,pm10,nitrogen_dioxide,ozone,carbon_monoxide,us_aqi",
            "start_date": start_date,
            "end_date":   end_date,
            "timezone":   TIMEZONE,
        },
        timeout=30,
    )
    aq_resp.raise_for_status()
    aq = aq_resp.json()["hourly"]

    print("  Fetching weather...")
    wx_resp = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude":   LAT,
            "longitude":  LON,
            "hourly":     "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
            "start_date": start_date,
            "end_date":   end_date,
            "timezone":   TIMEZONE,
        },
        timeout=30,
    )
    wx_resp.raise_for_status()
    wx = wx_resp.json()["hourly"]

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

    df = df.drop_duplicates(subset=['datetime'])

    for col in ['aqi', 'pm2_5', 'pm10', 'temperature', 'wind_speed']:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

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

    # Rain code
    def get_rain_code(precip):
        if precip == 0:    return 0
        elif precip < 2.5: return 1
        elif precip < 10:  return 2
        else:              return 3

    df['rain_code'] = df['precipitation'].apply(get_rain_code)

    # Lag features
    for lag in [1, 2, 3, 6, 12, 24, 48, 72]:
        df[f'lag{lag}'] = df['aqi'].shift(lag)

    # Rolling statistics
    df['aqi_ma6']   = df['aqi'].rolling(6).mean()
    df['aqi_ma12']  = df['aqi'].rolling(12).mean()
    df['aqi_ma24']  = df['aqi'].rolling(24).mean()
    df['aqi_std12'] = df['aqi'].rolling(12).std()

    # Trend features
    df['aqi_trend_3h']  = df['aqi'] - df['aqi'].shift(3)
    df['aqi_trend_6h']  = df['aqi'] - df['aqi'].shift(6)
    df['aqi_trend_24h'] = df['aqi'] - df['aqi'].shift(24)

    # Rolling min/max
    df['aqi_min_24h']   = df['aqi'].rolling(24).min()
    df['aqi_max_24h']   = df['aqi'].rolling(24).max()
    df['aqi_range_24h'] = df['aqi_max_24h'] - df['aqi_min_24h']

    # PM2.5 lag features
    df['pm25_lag24'] = df['pm2_5'].shift(24)
    df['pm25_ma12']  = df['pm2_5'].rolling(12).mean()

    # Cyclical hour encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Season encoding
    def get_season(month):
        if month in [12, 1, 2]:   return 0  # Winter
        elif month in [3, 4, 5]:  return 1  # Spring
        elif month in [6, 7, 8]:  return 2  # Summer
        else:                     return 3  # Autumn

    df['season']        = df['month'].apply(get_season)
    df['is_rush_hour']  = df['hour'].apply(lambda h: 1 if h in [7,8,9,17,18,19] else 0)
    df['is_smog_season']= df['month'].apply(lambda m: 1 if m in [11,12,1,2] else 0)

    # Target variables
    df['target_h24'] = df['aqi'].shift(-24)
    df['target_h48'] = df['aqi'].shift(-48)
    df['target_h72'] = df['aqi'].shift(-72)

    # Only drop NaN from feature columns — NOT from targets
    # This allows live rows to be stored even without future target values
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    print(f"   After feature engineering: {len(df)} rows")
    print(f"   Total columns: {len(df.columns)}")
    return df

# ============================================
# STEP 4: STORE TO MONGODB
# ============================================
def store_to_mongodb(df, collection):
    """Store processed features to MongoDB — skips duplicates"""

    store_cols = ['datetime'] + FEATURE_COLS + ['target_h24', 'target_h48', 'target_h72']
    df_store   = df[store_cols].copy()

    # Convert datetime to string
    df_store['datetime'] = df_store['datetime'].astype(str)

    # Replace NaN with None for MongoDB
    df_store = df_store.where(pd.notnull(df_store), None)

    records  = df_store.to_dict('records')
    inserted = 0
    skipped  = 0

    for record in records:
        try:
            collection.insert_one(record)
            inserted += 1
        except Exception:
            skipped += 1

    print(f"   Inserted: {inserted} new rows")
    print(f"    Skipped: {skipped} duplicate rows")
    return inserted

# ============================================
# STEP 5: UPDATE PENDING TARGETS
# ============================================
def update_pending_targets(collection):
    """
    Find rows with missing targets and fill them
    using lag1 of future rows that now exist in MongoDB
    """

    print("  Updating pending targets...")

    pending = list(collection.find(
        {"target_h24": None},
        {"_id": 1, "datetime": 1}
    ))

    updated = 0
    for doc in pending:
        dt = pd.to_datetime(doc['datetime'])

        t24 = collection.find_one({"datetime": str(dt + timedelta(hours=24))})
        t48 = collection.find_one({"datetime": str(dt + timedelta(hours=48))})
        t72 = collection.find_one({"datetime": str(dt + timedelta(hours=72))})

        update_fields = {}
        if t24: update_fields['target_h24'] = t24.get('lag1')
        if t48: update_fields['target_h48'] = t48.get('lag1')
        if t72: update_fields['target_h72'] = t72.get('lag1')

        if update_fields:
            collection.update_one(
                {"_id": doc['_id']},
                {"$set": update_fields}
            )
            updated += 1

    print(f"  Updated targets for {updated} rows")

# ============================================
# STEP 6: MAIN PIPELINE
# ============================================
def run_feature_pipeline(start_date=None, end_date=None):
    """
    Historical backfill:
        run_feature_pipeline("2024-01-01", "2025-12-31")

    Incremental (every 5 hours via GitHub Actions):
        run_feature_pipeline()
    """

    print("\n" + "="*60)
    print("FEATURE PIPELINE — ISLAMABAD AQI")
    print("="*60)

    if start_date is None:
        # fetch 4 days to ensure enough history for lag72
        end_dt     = datetime.now()
        start_dt   = end_dt - timedelta(days=4)
        start_date = start_dt.strftime("%Y-%m-%d")
        end_date   = end_dt.strftime("%Y-%m-%d")
        print(f"Mode      : INCREMENTAL (last 4 days with buffer)")
    else:
        print(f"Mode      : HISTORICAL BACKFILL")

    print(f"Date range: {start_date} → {end_date}")
    print("-"*60)

    print("\n[1/5] Connecting to MongoDB...")
    collection = get_mongo_collection()

    print("\n[2/5] Fetching raw data...")
    df_raw = fetch_raw_data(start_date, end_date)

    print("\n[3/5] Cleaning data...")
    df_clean = clean_data(df_raw)

    print("\n[4/5] Engineering features...")
    df_features = create_features(df_clean)

    print("\n[5/5] Storing to MongoDB...")
    store_to_mongodb(df_features, collection)

    print("\n[6/5] Updating pending targets...")
    update_pending_targets(collection)

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

    # ▶ HISTORICAL BACKFILL — uncomment to add more data
    #run_feature_pipeline("2024-01-01", "2026-05-30")

    # ▶ INCREMENTAL — runs every 5 hours via GitHub Actions
    run_feature_pipeline()