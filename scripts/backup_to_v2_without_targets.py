"""
Copy backup JSON to aqi_features_v2 collection, removing target columns.
This aligns with Plan A: separate features collection for live forecasting.
"""

import json
import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# MongoDB Connection
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME", "islamabad_aqi")
BACKUP_FILE = "backup/aqi_features_v1_backup.json"
TARGET_COLLECTION = "aqi_features_v2"

# Target columns to remove (Plan A: features only, no targets)
TARGET_COLUMNS = ["target_h24", "target_h48", "target_h72"]

def load_backup_json(filepath):
    """Load backup JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def remove_target_columns(record):
    """Remove target columns from a record"""
    cleaned_record = record.copy()
    for col in TARGET_COLUMNS:
        cleaned_record.pop(col, None)  # Remove if exists, no error if missing
    return cleaned_record

def copy_backup_to_v2():
    """Copy backup data to aqi_features_v2, removing targets"""
    try:
        # Connect to MongoDB
        client = MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        collection = db[TARGET_COLLECTION]
        
        # Load backup data
        print(f"Loading backup from: {BACKUP_FILE}")
        backup_data = load_backup_json(BACKUP_FILE)
        
        if not isinstance(backup_data, list):
            backup_data = [backup_data]
        
        print(f"Total records in backup: {len(backup_data)}")
        
        # Clean records (remove targets)
        cleaned_records = []
        for record in backup_data:
            cleaned_record = remove_target_columns(record)
            cleaned_records.append(cleaned_record)
        
        # Clear existing collection (optional - remove if you want to merge)
        print(f"Clearing existing {TARGET_COLLECTION} collection...")
        collection.delete_many({})
        
        # Insert cleaned records
        print(f"Inserting {len(cleaned_records)} records without target columns...")
        result = collection.insert_many(cleaned_records)
        
        print(f"\n✅ Success!")
        print(f"   - Inserted: {len(result.inserted_ids)} records")
        print(f"   - Collection: {TARGET_COLLECTION}")
        print(f"   - Target columns removed: {TARGET_COLUMNS}")
        
        # Verify
        count = collection.count_documents({})
        print(f"   - Total documents in {TARGET_COLLECTION}: {count}")
        
        # Check first record to verify no targets
        first_record = collection.find_one()
        if first_record:
            has_targets = any(col in first_record for col in TARGET_COLUMNS)
            print(f"   - Target columns present in collection: {has_targets}")
            if not has_targets:
                print("   ✓ All target columns successfully removed!")
        
        client.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    copy_backup_to_v2()
