from dagster_mlflow import mlflow_tracking
import os
from mlflow.tracking import MlflowClient
from cdsapi import Client
import dagster as dg

# Configuration for Local SQLite and Local Artifacts
# Using DAGSTER_HOME if set, otherwise, defaults to the current directory
# where the Dagster process is run.

BASE_DIR = os.path.abspath(os.getenv("DAGSTER_HOME", "."))  # base directory for all MLflow data.
SQLITE_DB_FILENAME = "mlflow_local_tracking.db"  # Name of the SQLite database file
# Construct full paths
SQLITE_DB_PATH = os.path.join(BASE_DIR, SQLITE_DB_FILENAME)

DEFAULT_EXPERIMENT_NAME = "era5_temperature_analysis"

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


class CDSAPI(dg.ConfigurableResource):
    host_url: str = "https://cds.climate.copernicus.eu/api"
    api_key: str = ""

    @property
    def client(self):
        return Client(url=self.host_url, key=self.api_key)
