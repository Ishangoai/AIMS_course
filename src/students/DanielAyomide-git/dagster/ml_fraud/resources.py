
import dagster as dg
from dagster_mlflow import mlflow_tracking
from mlflow.tracking import MlflowClient

from .configs import DataConfig, ExperimentConfig, MLFlowConfig, ModelConfig, ModelPromotionConfig

# Instantiate all configs
data_config = DataConfig()
model_config = ModelConfig()
mlflow_config = MLFlowConfig()
experiment_config = ExperimentConfig()
model_promotion_config = ModelPromotionConfig()

# Define the MLflow resource

# NOTE: This is the high-level MLFlow resource used for
# logging experiments, runs, metrics, params, and artifacts.
mlflow_resource = mlflow_tracking.configured(
    {
        # Point MLflow to use the local SQLite database
        "mlflow_tracking_uri": mlflow_config.tracking_uri,
        "experiment_name": experiment_config.experiment_name
    }
)


# Low-level MlflowClient for advanced API access (transition_model_version_stage,....)
@dg.resource
def mlflow_client(_):
    return MlflowClient(tracking_uri=mlflow_config.tracking_uri)
