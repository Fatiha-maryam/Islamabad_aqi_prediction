import os
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv()

client = MongoClient(os.environ['MONGODB_URI'])
collection = client['aqi_db']['aqi_features']

print("Total docs:", collection.count_documents({}))

latest = collection.find_one(sort=[("datetime", -1)])
oldest = collection.find_one(sort=[("datetime", 1)])
print("Latest datetime:", latest['datetime'])
print("Oldest datetime:", oldest['datetime'])