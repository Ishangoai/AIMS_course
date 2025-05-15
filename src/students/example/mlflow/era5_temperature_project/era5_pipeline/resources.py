from dagster_mlflow import mlflow_tracking

# Define the MLflow resource
mlflow_resource = mlflow_tracking.configured(
    {
        "experiment_name": "era5_temperature_analysis", # Name of the experiment in MLflow
        "mlflow_tracking_uri": "http://localhost:5000", # Default MLflow tracking URI
        # For local file-based tracking, you can also point to "file:./mlruns"
        # or omit if MLflow is already configured globally to use local tracking.
    }
)