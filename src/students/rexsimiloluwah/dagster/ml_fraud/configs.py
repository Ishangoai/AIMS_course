import os
from typing import List

import pydantic as pyd
from dagster import ConfigurableResource


class DataConfig(ConfigurableResource):
    """Data processing configuration.
    """
    dataset_url: str = pyd.Field(
        default="https://raw.githubusercontent.com/aduuna/Kaggle-Data-Credit-Card-Fraud-Detection/master/samplecreditcard.csv",
        description="URL to download the dataset from"
    )
    test_size: float = pyd.Field(
        default=0.2,
        description="Proportion of the dataset to include in the test split"
    )
    random_state: int = pyd.Field(
        default=42,
        description="Random seed for reproducibility"
    )


class ModelConfig(ConfigurableResource):
    """Model training configuration.
    """
    n_estimators: int = pyd.Field(
        default=500,
        description="Number of trees in the Random Forest"
    )
    max_depth: int = pyd.Field(
        default=30,
        description="Maximum depth of the tree"
    )
    max_depth_options: List[int] = pyd.Field(
        default=[3, 5, 8, 10, 12],
        description="Options for maximum depth during hyperparameter tuning"
    )
    min_samples_split: int = pyd.Field(
        default=2,
        description="Minimum number of samples required to split an internal node"
    )
    min_samples_leaf: int = pyd.Field(
        default=5,
        description="Minimum number of samples required to be at a leaf node"
    )
    random_state: int = pyd.Field(
        default=42,
        description="Random seed for reproducibility"
    )
    scoring_metric: str = pyd.Field(
        default="f1",
        description="Metric to optimize during hyperparameter tuning"
    )
    cv_folds: int = pyd.Field(
        default=3,
        description="Number of cross-validation folds"
    )
    feature_importance_threshold: float = pyd.Field(
        default=0.02,
        description="Threshold for feature importance to select features"
    )


class ModelPromotionConfig(ConfigurableResource):
    """Configuration for model promotion thresholds.
    F1 score and ROC-AUC thresholds for promoting models to Staging and Production.

    Why?
    F1 score is the harmonic mean of precision and recall, providing a balance between the two metrics.
    ROC-AUC measures the model's ability to distinguish between classes across all classification thresholds.
    """
    staging_f1_threshold: float = pyd.Field(
        default=0.75,
        description="Minimum F1-score threshold for promoting a model to Staging."
    )
    production_f1_threshold: float = pyd.Field(
        default=0.80,
        description="Minimum F1-score threshold for promoting a model to Production."
    )
    staging_roc_auc_threshold: float = pyd.Field(
        default=0.80,
        description="Minimum ROC-AUC threshold for promoting a model to Staging."
    )
    production_roc_auc_threshold: float = pyd.Field(
        default=0.85,
        description="Minimum ROC-AUC threshold for promoting a model to Production."
    )


class MLFlowConfig(ConfigurableResource):
    """Configuration for MLFlow."""

    base_dir: str = pyd.Field(
        default_factory=lambda: os.path.abspath(os.getenv("DAGSTER_HOME", ".")),
        description=(
            "Base directory for all MLflow data. Using DAGSTER_HOME if set, "
            "otherwise, defaults to the current directory"
        )
    )
    db_filename: str = pyd.Field(
        default="mlflow_local_tracking.db",
        description="Name of the SQLite database file for local tracking"
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
        default="ml_fraud_detection",
        description="Name of the experiment for this project"
    )
