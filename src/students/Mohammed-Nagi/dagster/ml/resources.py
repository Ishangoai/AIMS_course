import os

import dagster as dg
import dagster_slack
import numpy as np
import pandas as pd
import pydantic as pyd
import xarray as xr
from dagster_mlflow import mlflow_tracking
from mlflow.tracking import MlflowClient

# --- Central Path Definition ---
# This block defines the path to your MLflow database for all resources to use.
BASE_DIR = os.path.abspath(os.getenv("DAGSTER_HOME", "."))
SQLITE_DB_FILENAME = "mlflow_local_tracking.db"
SQLITE_DB_PATH = os.path.join(BASE_DIR, SQLITE_DB_FILENAME)


# --- Central Resource Definitions ---
# All assets in your project will refer to these single resource objects.

# MLflow resource for Dagster's logging integration
mlflow_resource = mlflow_tracking.configured(
    {
        "mlflow_tracking_uri": f"sqlite:///{SQLITE_DB_PATH}",
        "experiment_name": "fraud_detection_analysis",
    }
)


# MLflowClient resource for direct API access (e.g., promoting models)
@dg.resource
def mlflow_client(_):
    return MlflowClient(tracking_uri=f"sqlite:///{SQLITE_DB_PATH}")


# The one and only Slack resource for your project
slack_resource = dagster_slack.SlackResource(token=dg.EnvVar("SLACK_AIMS_COURSE_BOT_TOKEN"))


# --- Configuration Classes ---
# All Config classes are defined here for clarity.

# For the Fraud pipeline
class PromotionConfig(dg.Config):
    """Configuration for model promotion thresholds."""
    staging_f1_threshold: float = pyd.Field(
        default=0.85,
        description="Minimum acceptable F1 Score for promoting a model to Staging."
    )
    staging_accuracy_threshold: float = pyd.Field(
        default=0.99,
        description="Minimum acceptable Accuracy for promoting a model to Staging."
    )


# For the Weather pipeline
class Era5RequestConfig(dg.Config):
    product_type: str = pyd.Field(default="reanalysis")
    variable: str = pyd.Field(default="2m_temperature")
    year: str = pyd.Field(default="2023")
    month: str = pyd.Field(default="01")
    day: list[str] = pyd.Field(default=[f"{i:02d}" for i in range(1, 16)])
    time: list[str] = pyd.Field(default=["00:00", "06:00", "12:00", "18:00"])
    area: list[float] = pyd.Field(default=[50.0, -5.0, 45.0, 5.0])
    format: str = pyd.Field(default="netcdf")


class TuningConfig(dg.Config):
    max_hyperopt_evals: int = pyd.Field(default=20)


# --- Other Resources (e.g., for the weather pipeline) ---

class CDSAPIResource(dg.ConfigurableResource):
    """A mock resource for the CDS API to allow the weather pipeline to run."""
    host_url: str = "https://cds.climate.copernicus.eu/api"
    api_key: str = dg.EnvVar("CDS_API_KEY")

    @property
    def client(self):
        class MockCDSClient:
            def retrieve(self, *args, **kwargs):
                print("Mock CDS Client: Pretending to download data.")
                # Create a dummy NetCDF file for the asset to read
                dummy_ds = xr.Dataset(
                    {'t2m': (('time', 'latitude', 'longitude'), np.random.rand(10, 2, 2))},
                    coords={
                        'time': pd.to_datetime(pd.date_range('2023-01-01', periods=10)),
                        'latitude': [50, 45],
                        'longitude': [-5, 5]
                    }
                )
                dummy_ds.to_netcdf(args[2])
        return MockCDSClient()
