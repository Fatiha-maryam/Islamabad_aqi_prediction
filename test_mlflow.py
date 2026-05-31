import os
import mlflow
from dotenv import load_dotenv
load_dotenv()

uri      = os.environ.get("MLFLOW_TRACKING_URI", "")
username = os.environ.get("MLFLOW_TRACKING_USERNAME", "")
password = os.environ.get("MLFLOW_TRACKING_PASSWORD", "")

print(f"URI: {uri}")
print(f"Username: {username}")
print(f"Password length: {len(password)}")

# Explicitly set all three
os.environ["MLFLOW_TRACKING_URI"]      = uri
os.environ["MLFLOW_TRACKING_USERNAME"] = username
os.environ["MLFLOW_TRACKING_PASSWORD"] = password

mlflow.set_tracking_uri(uri)

try:
    mlflow.set_experiment("Islamabad_AQI_Prediction")
    with mlflow.start_run(run_name="test_run"):
        mlflow.log_metric("test_metric", 1.0)
    print(" Full MLflow test passed!")
except Exception as e:
    print(f" Error: {e}")