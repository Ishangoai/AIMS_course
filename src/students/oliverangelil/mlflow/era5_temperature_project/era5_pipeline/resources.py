import os

from dagster_mlflow import mlflow_tracking

# Configuration for Local SQLite and Local Artifacts
# Using DAGSTER_HOME if set, otherwise, defaults to the current directory
# where the Dagster process is run.

BASE_DIR = os.path.abspath(os.getenv("DAGSTER_HOME", "."))  # base directory for all MLflow data.
SQLITE_DB_FILENAME = "mlflow_local_tracking.db"  # Name of the SQLite database file
# LOCAL_ARTIFACTS_DIRNAME = "mlartifacts_local_store"  # Name of the local artifacts directory

# Construct full paths
SQLITE_DB_PATH = os.path.join(BASE_DIR, SQLITE_DB_FILENAME)
# LOCAL_ARTIFACT_ROOT_PATH = os.path.join(BASE_DIR, LOCAL_ARTIFACTS_DIRNAME)

# Ensure the local artifact root directory exists
# os.makedirs(LOCAL_ARTIFACT_ROOT_PATH, exist_ok=True)

DEFAULT_EXPERIMENT_NAME = "era5_temperature_analysis"

# Construct the artifact location for this specific default experiment.
# DEFAULT_EXPERIMENT_ARTIFACT_LOCATION = f"file:{os.path.join(LOCAL_ARTIFACT_ROOT_PATH, DEFAULT_EXPERIMENT_NAME)}"
# Ensure this specific experiment artifact directory also exists
# os.makedirs(os.path.join(LOCAL_ARTIFACT_ROOT_PATH, DEFAULT_EXPERIMENT_NAME), exist_ok=True)

# Define the MLflow resource
mlflow_resource = mlflow_tracking.configured(
    {
        # Point MLflow to use the local SQLite database
        "mlflow_tracking_uri": f"sqlite:///{SQLITE_DB_PATH}",
        "experiment_name": DEFAULT_EXPERIMENT_NAME,
        # Configuration for creating the experiment if it doesn't exist.
        # "artifact_location": DEFAULT_EXPERIMENT_ARTIFACT_LOCATION,

        # "save_lifecycle_data": True,

        # "mlflow_tracking_uri": "http://localhost:5000",
        # For local file-based tracking, you can also point to "file:./mlruns"
        # or omit if MLflow is already configured globally to use local tracking.
    }
)
