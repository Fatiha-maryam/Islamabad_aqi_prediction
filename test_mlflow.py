import os
import dagshub
import mlflow

# --- IMPORTANT: Replace these with your actual credentials for this test ---
os.environ["MLFLOW_TRACKING_USERNAME"] = "Fatiha-maryam"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "797e3df3a4c2d01d9c002788c93d9a7990c18257"

# Initialize DagsHub and MLflow
dagshub.init(repo_owner="Fatiha-maryam", repo_name="Islamabad_aqi_prediction", mlflow=True)

# Test the connection by setting an experiment
mlflow.set_experiment("connection_test")
print(" Authentication successful! Local test passed.")