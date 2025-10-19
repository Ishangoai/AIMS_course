import os

from dagster import ConfigurableResource
from dagster_mlflow import mlflow_tracking


class FraudDataConfig(ConfigurableResource):
    """Configuration for fraud detection data source"""
    data_source: str = "https://raw.githubusercontent.com/aduuna/Kaggle-Data-Credit-Card-Fraud-Detection/master/samplecreditcard.csv"


# === NEW: Define local MLflow storage directory ===
# This sets MLflow DB + artifacts inside the ml_fraud folder
MLFLOW_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "mlflow_artifacts"
))
FRAUD_SQLITE_DB_PATH = os.path.join(MLFLOW_DIR, "mlflow_fraud_tracking.db")

# Ensure directory exists
os.makedirs(MLFLOW_DIR, exist_ok=True)

# === Dagster-MLflow resource ===
mlflow_fraud_resource = mlflow_tracking.configured({
    "mlflow_tracking_uri": f"sqlite:///{FRAUD_SQLITE_DB_PATH}",
    "experiment_name": "fraud_detection_pipeline",
})
