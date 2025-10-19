"""Dagster MLflow resource setup file.

This module initializes MLflow tracking and resource clients used throughout
the Dagster pipeline. It pulls configuration values from corresponding config
classes, providing both high-level and low-level MLflow interfaces:

- `mlflow_resource`: Used by Dagster assets and ops for standard experiment
  logging (parameters, metrics, artifacts).
- `mlflow_client`: A lower-level MlflowClient resource for programmatic access
  to the MLflow REST API (e.g., managing model versions or transitions).
"""

import dagster as dg
from dagster_mlflow import mlflow_tracking
from mlflow.tracking import MlflowClient

from .configs import (
    DataConfig,
    ExperimentConfig,
    MLFlowConfig,
    ModelConfig,
    ModelPromotionConfig,
)

# Instantiate all configuration objects
data_config = DataConfig()
model_config = ModelConfig()
mlflow_config = MLFlowConfig()
experiment_config = ExperimentConfig()
model_promotion_config = ModelPromotionConfig()

# Define the MLflow resource
# NOTE: This is the high-level MLflow resource used by Dagster for
# experiment logging (runs, metrics, parameters, artifacts, etc.).
mlflow_resource = mlflow_tracking.configured(
    {
        "mlflow_tracking_uri": mlflow_config.tracking_uri,
        "experiment_name": experiment_config.experiment_name,
    }
)


@dg.resource
def mlflow_client(_) -> MlflowClient:
    """Provides a low-level MLflow client resource.

    This resource allows direct access to MLflow’s tracking API for operations
    not covered by the high-level `mlflow_tracking` resource, such as:
    - Transitioning model versions between stages
    - Querying model registry entries
    - Deleting experiments or runs

    Args:
        _: The Dagster resource context (unused).

    Returns:
        MlflowClient: A configured MLflow client instance.
    """
    return MlflowClient(tracking_uri=mlflow_config.tracking_uri)
