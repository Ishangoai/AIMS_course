import os

import dagster as dg
import pydantic as pyd
from dagster_mlflow import mlflow_tracking
from mlflow.tracking import MlflowClient

BASE_DIR = os.path.abspath(os.getenv("DAGSTER_HOME", "."))
SQLITE_DB_FILENAME = "mlflow_local_tracking.db"
# Construct full paths
SQLITE_DB_PATH = os.path.join(BASE_DIR, SQLITE_DB_FILENAME)

DEFAULT_EXPERIMENT_NAME = "fraud_detection_ml"


# Define the MLflow resource for fraud detection
mlflow_resource = mlflow_tracking.configured(
    {
        "mlflow_tracking_uri": f"sqlite:///{SQLITE_DB_PATH}",
        "experiment_name": DEFAULT_EXPERIMENT_NAME,
    }
)


# Raw MlflowClient for advanced API access
@dg.resource
def mlflow_client(_):
    return MlflowClient(tracking_uri=f"sqlite:///{SQLITE_DB_PATH}")


class FraudModelConfig(dg.Config):
    """Configuration for fraud detection model hyperparameter tuning"""

    n_estimators: list[int] = pyd.Field(
        default=[50, 100, 200, 300, 500], description="List of n_estimators values to try during hyperparameter tuning"
    )
    cv_folds: int = pyd.Field(default=3, description="Number of cross-validation folds")
    random_state: int = pyd.Field(default=42, description="Random state for reproducibility")
