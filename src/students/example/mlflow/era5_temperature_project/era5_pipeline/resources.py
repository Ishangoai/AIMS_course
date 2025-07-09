from dagster_mlflow import mlflow_tracking
import os
from mlflow.tracking import MlflowClient
from cdsapi import Client
import dagster as dg
import pydantic as pyd

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


# configuration for the raw_netcdf_dataset asset
class Era5RequestConfig(dg.Config):
    product_type: str = pyd.Field(
        default="reanalysis",
        description="The product type to request"
    )
    variable: str = pyd.Field(
        default="2m_temperature",
        description="The meteorological variable to retrieve"
    )
    year: str = pyd.Field(
        default="2023",
        description="The year for which to retrieve data"
    )
    month: str = pyd.Field(
        default="01",
        description="The month for which to retrieve data"
    )
    day: list[str] = pyd.Field(
        default=[f"{i:02d}" for i in range(1, 16)],
        description="A list of days to retrieve"
    )
    time: list[str] = pyd.Field(
        default=["00:00", "06:00", "12:00", "18:00"],
        description="Times of day to retrieve data"
    )
    area: list[float] = pyd.Field(
        default=[50.0, -5.0, 45.0, 5.0],
        description="Area: [North, West, South, East]"
    )
    format: str = pyd.Field(
        default="netcdf",
        description="Format to download (e.g., netcdf)"
    )
