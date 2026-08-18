"""
Training Data Preparation Pipeline
Computes targets for all historical features from aqi_features_v2
Runs: Every Sunday 1:00 AM (before training_pipeline)
Output: Updates aqi_training_data with features + complete targets
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME", "islamabad_aqi")
FEATURES_COLLECTION = "aqi_features_v2"
TRAINING_COLLECTION = "aqi_training_data"

def connect_mongodb():
    """Connect to MongoDB"""
    if not MONGODB_URI:
        raise ValueError("MONGODB_URI environment variable not set!")
    
    client = MongoClient(
        MONGODB_URI,
        serverSelectionTimeoutMS=60000,
        connectTimeoutMS=60000
    )
    return client[DB_NAME]

def load_all_features():
    """Load all data from aqi_features_v2"""
    print("\n[1/4] Loading all features from aqi_features_v2...")
    
    db = connect_mongodb()
    features_col = db[FEATURES_COLLECTION]
    
    cursor = features_col.find({}, {"_id": 0})
    df = pd.DataFrame(list(cursor))
    
    if len(df) == 0:
        print(" No data found in aqi_features_v2!")
        return None
    
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    print(f"   Loaded {len(df)} rows")
    print(f"   Date range: {df['datetime'].min()} → {df['datetime'].max()}")
    
    return df

def compute_targets(df):
    """
    Compute target columns (target_h24, target_h48, target_h72) from actual future AQI values
    
    For each row at time t:
    - target_h24 = AQI value at t+24h
    - target_h48 = AQI value at t+48h
    - target_h72 = AQI value at t+72h
    """
    print("\n[2/4] Computing targets from historical AQI values...")
    
    # Create mapping of datetime → lag1 value (which represents AQI at that hour)
    datetime_to_aqi = dict(zip(df['datetime'], df['lag1']))
    
    targets_h24 = []
    targets_h48 = []
    targets_h72 = []
    
    for idx, row in df.iterrows():
        current_dt = row['datetime']
        
        # Look up actual AQI at future times
        dt_24h = current_dt + timedelta(hours=24)
        dt_48h = current_dt + timedelta(hours=48)
        dt_72h = current_dt + timedelta(hours=72)
        
        # Get AQI values if they exist
        target_24 = datetime_to_aqi.get(dt_24h)
        target_48 = datetime_to_aqi.get(dt_48h)
        target_72 = datetime_to_aqi.get(dt_72h)
        
        targets_h24.append(target_24)
        targets_h48.append(target_48)
        targets_h72.append(target_72)
    
    df['target_h24'] = targets_h24
    df['target_h48'] = targets_h48
    df['target_h72'] = targets_h72
    
    print(f"   Target columns computed")
    print(f"   target_h24: {df['target_h24'].notna().sum()} values")
    print(f"   target_h48: {df['target_h48'].notna().sum()} values")
    print(f"   target_h72: {df['target_h72'].notna().sum()} values")
    
    return df

def filter_complete_rows(df):
    """
    Keep only rows with ALL target values present.
    Excludes last ~72 hours (future AQI not yet known).
    """
    print("\n[3/4] Filtering to keep only complete rows...")
    
    total_rows = len(df)
    
    # Keep only rows where ALL targets exist
    df_complete = df.dropna(subset=['target_h24', 'target_h48', 'target_h72'])
    
    rows_removed = total_rows - len(df_complete)
    
    print(f"   Total rows: {total_rows}")
    print(f"   Rows with complete targets: {len(df_complete)}")
    print(f"   Rows excluded (last ~72h): {rows_removed}")
    
    if len(df_complete) == 0:
        print("   ⚠️  WARNING: No complete rows found!")
        return None
    
    print(f"   ✓ Ready for training: {len(df_complete)} complete rows")
    
    return df_complete

def save_to_training_collection(df):
    """
    Clear aqi_training_data and write fresh data with targets.
    """
    print("\n[4/4] Saving to aqi_training_data...")
    
    db = connect_mongodb()
    training_col = db[TRAINING_COLLECTION]
    
    # Clear existing data
    print("   Clearing existing data...")
    training_col.delete_many({})
    
    # Prepare records for insertion
    # Keep datetime as string for MongoDB consistency
    df_to_insert = df.copy()
    df_to_insert['datetime'] = df_to_insert['datetime'].astype(str)
    df_to_insert = df_to_insert.where(pd.notnull(df_to_insert), None)
    
    records = df_to_insert.to_dict('records')
    
    # Insert records
    print(f"   Inserting {len(records)} records...")
    result = training_col.insert_many(records)
    
    print(f"\n✅ SUCCESS!")
    print(f"   Inserted: {len(result.inserted_ids)} records")
    print(f"   Collection: {TRAINING_COLLECTION}")
    print(f"   All records have complete targets: ['target_h24', 'target_h48', 'target_h72']")
    
    # Verify
    count = training_col.count_documents({})
    print(f"   Total documents in {TRAINING_COLLECTION}: {count}")
    
    # Sample verification
    sample = training_col.find_one()
    if sample:
        print(f"\n   Sample record:")
        print(f"     datetime: {sample.get('datetime', 'N/A')}")
        print(f"     target_h24: {sample.get('target_h24', 'MISSING')}")
        print(f"     target_h48: {sample.get('target_h48', 'MISSING')}")
        print(f"     target_h72: {sample.get('target_h72', 'MISSING')}")

def run_training_data_prep():
    """Main training data preparation pipeline"""
    
    print("\n" + "="*70)
    print("TRAINING DATA PREPARATION PIPELINE")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Runs: Every Sunday 1:00 AM (before training_pipeline)")
    print(f"Purpose: Compute targets from aqi_features_v2 → update aqi_training_data")
    print("="*70)
    
    try:
        # Step 1: Load all features
        df = load_all_features()
        if df is None or len(df) == 0:
            print("\n❌ FAILED: No data to process")
            return False
        
        # Step 2: Compute targets
        df = compute_targets(df)
        
        # Step 3: Filter complete rows
        df_complete = filter_complete_rows(df)
        if df_complete is None or len(df_complete) == 0:
            print("\n❌ FAILED: No complete rows to save")
            return False
        
        # Step 4: Save to training collection
        save_to_training_collection(df_complete)
        
        print("\n" + "="*70)
        print("TRAINING DATA PREPARATION COMPLETED SUCCESSFULLY")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("Next: training_pipeline.py will run in 1 hour")
        print("="*70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_training_data_prep()
    exit(0 if success else 1)
