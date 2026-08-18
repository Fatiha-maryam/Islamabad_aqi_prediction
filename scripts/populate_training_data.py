"""
Step 2: Copy backup to aqi_training_data with clean targets
Removes rows with incomplete/missing target columns for training-only data
"""

import json
import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME", "islamabad_aqi")
BACKUP_FILE = "backup/aqi_features_v1_backup.json"
TRAINING_COLLECTION = "aqi_training_data"

# Target columns that MUST be present and complete
TARGET_COLUMNS = ["target_h24", "target_h48", "target_h72"]

def load_backup_json(filepath):
    """Load backup JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def filter_complete_rows(records):
    """
    Filter records to keep only those with complete target columns.
    Drops rows where any target is NaN, None, or missing.
    """
    df = pd.DataFrame(records)
    
    print(f"\n📊 Data Quality Check:")
    print(f"   Total rows in backup: {len(df)}")
    
    # Show which rows have missing targets
    for col in TARGET_COLUMNS:
        missing_count = df[col].isna().sum()
        print(f"   {col}: {missing_count} missing values")
    
    # Keep only rows where ALL targets are present and not NaN
    df_clean = df.dropna(subset=TARGET_COLUMNS)
    
    rows_dropped = len(df) - len(df_clean)
    print(f"\n   Rows dropped (incomplete targets): {rows_dropped}")
    print(f"   Rows kept (complete targets): {len(df_clean)}")
    
    if len(df_clean) == 0:
        print("\n   ⚠️  WARNING: No complete rows found!")
        print("   All rows have missing target values.")
        return []
    
    # Convert back to list of dicts
    return df_clean.to_dict('records')

def copy_backup_to_training():
    """Copy complete rows from backup to aqi_training_data"""
    try:
        # Load backup data
        print(f"Loading backup from: {BACKUP_FILE}")
        backup_data = load_backup_json(BACKUP_FILE)
        
        if not isinstance(backup_data, list):
            backup_data = [backup_data]
        
        # Filter to keep only complete rows
        clean_records = filter_complete_rows(backup_data)
        
        if not clean_records:
            print("\n❌ No complete records to insert!")
            return
        
        # Connect to MongoDB
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[TRAINING_COLLECTION]
        
        # Check if collection has existing data
        existing_count = collection.count_documents({})
        if existing_count > 0:
            print(f"\n⚠️  Collection already has {existing_count} documents")
            response = input("Clear existing data? (y/n): ").strip().lower()
            if response == 'y':
                collection.delete_many({})
                print("   ✓ Cleared existing data")
            else:
                print("   ✓ Keeping existing data")
        
        # Insert clean records
        print(f"\nInserting {len(clean_records)} complete records...")
        result = collection.insert_many(clean_records)
        
        print(f"\n✅ Success!")
        print(f"   - Inserted: {len(result.inserted_ids)} records")
        print(f"   - Database: {DB_NAME}")
        print(f"   - Collection: {TRAINING_COLLECTION}")
        print(f"   - All records have complete targets: {TARGET_COLUMNS}")
        
        # Verify
        count = collection.count_documents({})
        print(f"   - Total documents in {TRAINING_COLLECTION}: {count}")
        
        # Sample verification
        sample = collection.find_one()
        if sample:
            print(f"\n📋 Sample record (first row):")
            print(f"   _id: {sample['_id']}")
            print(f"   datetime: {sample.get('datetime', 'N/A')}")
            print(f"   target_h24: {sample.get('target_h24', 'MISSING')}")
            print(f"   target_h48: {sample.get('target_h48', 'MISSING')}")
            print(f"   target_h72: {sample.get('target_h72', 'MISSING')}")
            
            # Count features
            feature_count = len([k for k in sample.keys() if k not in ['_id', 'datetime', 'target_h24', 'target_h48', 'target_h72']])
            print(f"   Feature columns: {feature_count}")
        
        client.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    copy_backup_to_training()
