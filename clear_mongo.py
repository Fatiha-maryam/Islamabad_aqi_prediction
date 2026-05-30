# save as clear_mongo.py
import os
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv()

client = MongoClient(os.environ['MONGODB_URI'])
result = client['aqi_db']['aqi_features'].delete_many({})
print(f"Deleted {result.deleted_count} documents")