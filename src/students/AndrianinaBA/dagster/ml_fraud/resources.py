import os

import dagster as dg
import pydantic as pyd

# Configuration for Local SQLite and Local Artifacts
# Using DAGSTER_HOME if set, otherwise, defaults to the current directory
# where the Dagster process is run.

BASE_DIR = os.path.abspath(os.getenv("DAGSTER_HOME", "."))  # base directory for all MLflow data.
SQLITE_DB_FILENAME = "mlflow_local_tracking_fraud.db"  # Name of the SQLite database file
# Construct full paths
SQLITE_DB_PATH = os.path.join(BASE_DIR, SQLITE_DB_FILENAME)

DEFAULT_EXPERIMENT_NAME = "ml_fraud_detection"


class FraudDataConfig(dg.ConfigurableResource):
    data_url: str = "https://raw.githubusercontent.com/aduuna/Kaggle-Data-Credit-Card-Fraud-Detection/master/samplecreditcard.csv"


fraud_data_source = FraudDataConfig()


class TuningConfig(dg.Config):
    """Configuration for model tuning and optimization"""
    max_hyperopt_evals: int = pyd.Field(
        default=20,
        description="Maximum number of evaluations allowed by Hyperopt during model tuning."
    )
