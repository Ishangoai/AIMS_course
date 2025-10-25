import os

import dagster as dg
import pydantic as pyd
from dagster_mlflow import mlflow_tracking
from mlflow.tracking import MlflowClient

# Configuration for Local SQLite and Local Artifacts
# Using DAGSTER_HOME if set, otherwise, defaults to the current directory
# where the Dagster process is run.

BASE_DIR = os.path.abspath(os.getenv("DAGSTER_HOME", "."))  # base directory for all MLflow data.
SQLITE_DB_FILENAME = "mlflow_local_tracking.db"  # Name of the SQLite database file
# Construct full paths
SQLITE_DB_PATH = os.path.join(BASE_DIR, SQLITE_DB_FILENAME)

DEFAULT_EXPERIMENT_NAME = "fraud_detection_pipeline"


# Define the MLflow resource
mlflow_resource = mlflow_tracking.configured(
    {
        # Point MLflow to use the local SQLite database
        "mlflow_tracking_uri": f"sqlite:///{SQLITE_DB_PATH}",
        "experiment_name": DEFAULT_EXPERIMENT_NAME,
    }
)


# Raw MlflowClient for advanced API access (transition_model_version_stage,....)
@dg.resource
def mlflow_client(_):
    return MlflowClient(tracking_uri=f"sqlite:///{SQLITE_DB_PATH}")


class FraudDataAPI(dg.ConfigurableResource):
    """Simple configurable resource that exposes a CSV URL for the dataset."""

    fraud_data_api: str = (
        "https://raw.githubusercontent.com/aduuna/Kaggle-Data-Credit-Card-Fraud-Detection/master/samplecreditcard.csv"
    )

    @property
    def url(self) -> str:
        return self.fraud_data_api


# ResourceDefinition for the data API
data_api_resource = FraudDataAPI()


class TuningConfig(dg.Config):
    """Configuration for model tuning and optimization"""

    max_hyperopt_evals: int = pyd.Field(
        default=20, description="Maximum number of evaluations allowed by Hyperopt during model tuning."
    )


class PromotionConfig(dg.Config):
    """Configuration for model promotion thresholds"""

    staging_mse_threshold: float = pyd.Field(
        default=1.5, description="Maximum acceptable MSE for promoting a model to Staging."
    )
    staging_mae_threshold: float = pyd.Field(
        default=1.5, description="Maximum acceptable MAE for promoting a model to Staging."
    )
    staging_r2_threshold: float = pyd.Field(
        default=0.8, description="Minimum acceptable R2 for promoting a model to Staging."
    )
