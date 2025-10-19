import os
from typing import List

import pydantic as pyd
from dagster import ConfigurableResource


class DataConfig(ConfigurableResource):
    """Configuration for data loading and preprocessing.

    This resource defines parameters for dataset retrieval and
    train-test split behavior used throughout the data pipeline.
    """

    dataset_url: str = pyd.Field(
        default=(
            "https://raw.githubusercontent.com/aduuna/Kaggle-Data-Credit-Card-Fraud-Detection/"
            "master/samplecreditcard.csv"
        ),
        description="URL to download the dataset from.",
    )
    test_size: float = pyd.Field(
        default=0.2,
        description="Proportion of the dataset to include in the test split (0–1).",
    )
    random_state: int = pyd.Field(
        default=40,
        description="Random seed for reproducibility.",
    )


class ModelConfig(ConfigurableResource):
    """Configuration for model training and hyperparameter tuning.

    This resource defines parameters that govern how the model is
    trained, evaluated, and tuned, including the number of estimators,
    depth, and cross-validation setup.
    """

    n_estimators: int = pyd.Field(
        default=161,
        description="Number of trees in the Random Forest ensemble.",
    )
    max_depth: int = pyd.Field(
        default=15,
        description="Maximum depth of each decision tree.",
    )
    max_depth_options: List[int] = pyd.Field(
        default_factory=lambda: [5, 10, 20, 30],
        description="Options for maximum depth during hyperparameter tuning.",
    )
    random_state: int = pyd.Field(
        default=40,
        description="Random seed for reproducibility.",
    )
    scoring_metric: str = pyd.Field(
        default="f1",
        description="Metric to optimize during hyperparameter tuning.",
    )
    cv_folds: int = pyd.Field(
        default=3,
        description="Number of cross-validation folds.",
    )


class ModelPromotionConfig(ConfigurableResource):
    """Configuration thresholds for promoting models to staging or production.

    Promotion is determined based on minimum acceptable F1-score and
    ROC-AUC thresholds. F1-score balances precision and recall, while
    ROC-AUC measures the model's ability to distinguish between classes.

    Attributes:
        staging_f1_threshold: Minimum F1-score for model promotion to staging.
        production_f1_threshold: Minimum F1-score for model promotion to production.
        staging_roc_auc_threshold: Minimum ROC-AUC for model promotion to staging.
        production_roc_auc_threshold: Minimum ROC-AUC for model promotion to production.
    """

    staging_f1_threshold: float = pyd.Field(
        default=0.80,
        description="Minimum F1-score threshold for promoting a model to staging.",
    )
    production_f1_threshold: float = pyd.Field(
        default=0.80,
        description="Minimum F1-score threshold for promoting a model to production.",
    )
    staging_roc_auc_threshold: float = pyd.Field(
        default=0.80,
        description="Minimum ROC-AUC threshold for promoting a model to staging.",
    )
    production_roc_auc_threshold: float = pyd.Field(
        default=0.85,
        description="Minimum ROC-AUC threshold for promoting a model to production.",
    )


class MLFlowConfig(ConfigurableResource):
    """Configuration for MLflow tracking setup.

    This resource defines the base directory and SQLite database file
    used for MLflow tracking. The tracking URI is generated dynamically.

    Attributes:
        base_dir: Base directory for MLflow tracking data.
        db_filename: Name of the SQLite file for MLflow tracking.
    """

    base_dir: str = pyd.Field(
        default_factory=lambda: os.path.abspath(os.getenv("DAGSTER_HOME", ".")),
        description=(
            "Base directory for all MLflow data. Uses DAGSTER_HOME if set; "
            "defaults to the current working directory otherwise."
        ),
    )
    db_filename: str = pyd.Field(
        default="mlflow_local_tracking.db",
        description="Name of the SQLite database file for MLflow local tracking.",
    )

    @property
    def tracking_uri(self) -> str:
        """Return the SQLite tracking URI for MLflow.

        Returns:
            str: SQLite connection string formatted as a MLflow tracking URI.
        """
        sqlite_db_path = os.path.join(self.base_dir, self.db_filename)
        return f"sqlite:///{os.path.abspath(sqlite_db_path)}"


class ExperimentConfig(ConfigurableResource):
    """Configuration for ML experiment management.

    This resource defines the experiment name used by MLflow
    and other pipeline components.
    """

    experiment_name: str = pyd.Field(
        default="ml_fraud_detection",
        description="Name of the experiment for this ML pipeline.",
    )
