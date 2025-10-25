import os

import dagster as dg
from dagster_mlflow import mlflow_tracking
from mlflow.tracking import MlflowClient

BASE_DIR = os.path.abspath(os.getenv("DAGSTER_HOME", "."))  # base directory for all MLflow data.
SQLITE_DB_FILENAME = "mlflow_local_tracking.db"  # Name of the SQLite database file
# Construct full paths
SQLITE_DB_PATH = os.path.join(BASE_DIR, SQLITE_DB_FILENAME)


mlflow_resource = mlflow_tracking.configured({
    "experiment_name": "fraud_detection_experiment",
    "mlflow_tracking_uri": f"sqlite:///{SQLITE_DB_PATH}"
})


# Raw MlflowClient for advanced API access (transition_model_version_stage,....)
@dg.resource
def mlflow_client(_):
    return MlflowClient(tracking_uri=f"sqlite:///{SQLITE_DB_PATH}")
