import os
from typing import List

import cdsapi
import dagster as dg
import pydantic as pyd
from dagster import ConfigurableResource
from dagster_mlflow import mlflow_tracking
from mlflow.tracking import MlflowClient

# from .configs import DataConfig, ExperimentConfig, MLFlowConfig, ModelConfig, ModelPromotionConfig

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


class CDSAPIResource(dg.ConfigurableResource):
    host_url: str = "https://cds.climate.copernicus.eu/api"
    api_key: str = dg.EnvVar("CDS_API_KEY")

    @property
    def client(self) -> cdsapi.Client:
        return cdsapi.Client(url=self.host_url, key=self.api_key)


# configuration for the raw_xarray_dataset asset
class Era5RequestConfig(dg.Config):
    product_type: str = pyd.Field(default="reanalysis", description="The product type to request")
    variable: str = pyd.Field(default="2m_temperature", description="The meteorological variable to retrieve")
    year: str = pyd.Field(default="2023", description="The year for which to retrieve data")
    month: str = pyd.Field(default="01", description="The month for which to retrieve data")
    day: list[str] = pyd.Field(default=[f"{i:02d}" for i in range(1, 16)], description="A list of days to retrieve")
    time: list[str] = pyd.Field(
        default=["00:00", "06:00", "12:00", "18:00"], description="Times of day to retrieve data"
    )
    area: list[float] = pyd.Field(default=[50.0, -5.0, 45.0, 5.0], description="Area: [North, West, South, East]")
    format: str = pyd.Field(default="netcdf", description="Format to download (e.g., netcdf)")


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


class DataConfig(ConfigurableResource):
    """Data processing configuration."""

    dataset_url: str = pyd.Field(
        default="https://raw.githubusercontent.com/aduuna/Kaggle-Data-Credit-Card-Fraud-Detection/master/samplecreditcard.csv",
        description="URL to download the dataset from",
    )
    test_size: float = pyd.Field(default=0.2, description="Proportion of the dataset to include in the test split")
    random_state: int = pyd.Field(default=42, description="Random seed for reproducibility")


class ModelConfig(ConfigurableResource):
    """Model training configuration."""

    n_estimators: int = pyd.Field(default=100, description="Number of trees in the Random Forest")
    max_depth: int = pyd.Field(default=10, description="Maximum depth of the tree")
    max_depth_options: List[int] = pyd.Field(
        default=[5, 10, 15, 20], description="Options for maximum depth during hyperparameter tuning"
    )
    random_state: int = pyd.Field(default=42, description="Random seed for reproducibility")
    scoring_metric: str = pyd.Field(default="f1", description="Metric to optimize during hyperparameter tuning")
    cv_folds: int = pyd.Field(default=3, description="Number of cross-validation folds")


class ModelPromotionConfig(ConfigurableResource):
    """Configuration for model promotion thresholds.
    F1 score and ROC-AUC thresholds for promoting models to Staging and Production.

    Why?
    F1 score is the harmonic mean of precision and recall, providing a balance between the two metrics.
    ROC-AUC measures the model's ability to distinguish between classes across all classification thresholds.
    """

    staging_f1_threshold: float = pyd.Field(
        default=0.75, description="Minimum F1-score threshold for promoting a model to Staging."
    )
    production_f1_threshold: float = pyd.Field(
        default=0.80, description="Minimum F1-score threshold for promoting a model to Production."
    )
    staging_roc_auc_threshold: float = pyd.Field(
        default=0.80, description="Minimum ROC-AUC threshold for promoting a model to Staging."
    )
    production_roc_auc_threshold: float = pyd.Field(
        default=0.85, description="Minimum ROC-AUC threshold for promoting a model to Production."
    )


class MLFlowConfig(ConfigurableResource):
    """Configuration for MLFlow."""

    base_dir: str = pyd.Field(
        default_factory=lambda: os.path.abspath(os.getenv("DAGSTER_HOME", ".")),
        description=(
            "Base directory for all MLflow data. Using DAGSTER_HOME if set, "
            "otherwise, defaults to the current directory"
        ),
    )
    db_filename: str = pyd.Field(
        default="mlflow_local_tracking.db", description="Name of the SQLite database file for local tracking"
    )

    @property
    def tracking_uri(self) -> str:
        """Construct the SQLite tracking URI from base_dir and db_filename."""
        sqlite_db_path = os.path.join(self.base_dir, self.db_filename)
        # Ensure the path is absolute for the URI
        sqlite_db_path = os.path.abspath(sqlite_db_path)
        return f"sqlite:///{sqlite_db_path}"


class ExperimentConfig(ConfigurableResource):
    """Configuration for the experiment."""

    experiment_name: str = pyd.Field(
        default="ml_fraud_detection", description="Name of the experiment for this project"
    )


data_cgnfig = DataConfig()
model_config = ModelConfig()
mlflow_config = MLFlowConfig()
experiment_config = ExperimentConfig()
model_promotion_config = ModelPromotionConfig()
