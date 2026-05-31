import os
import mlflow
from dotenv import load_dotenv
load_dotenv()

username = os.environ.get("MLFLOW_TRACKING_USERNAME")
password = os.environ.get("MLFLOW_TRACKING_PASSWORD")

print(f"Username: {username}")
print(f"Password exists: {bool(password)}")

# Set tracking URI with credentials embedded
tracking_uri = f"https://{username}:{password}@dagshub.com/{username}/Islamabad_aqi_prediction.mlflow"
mlflow.set_tracking_uri(tracking_uri)

# Test connection
try:
    mlflow.set_experiment("test_connection")
    print("MLflow connected successfully!")
except Exception as e:
    print(f" Error: {e}")