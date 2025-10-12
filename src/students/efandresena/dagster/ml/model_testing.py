import os

import mlflow
import pandas as pd

MODEL_NAME = "tuned-temp-forecaster"
MODEL_STAGE = "Production"

# Construct an absolute path to the database file.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "..", "..", "..", "..", "..", "mlflow_local_tracking.db")

# Construct the URI to load the model from the registry
model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

# Set up MLflow Tracking
mlflow.set_tracking_uri(f"sqlite:///{os.path.abspath(DB_PATH)}")

# Prepare Input Data
input_data = pd.DataFrame(data={"t2m_celsius_lag1": [5.0, 10.0, 15.5, 20.2]})

# Run Prediction
results = mlflow.models.predict(  # type: ignore
    model_uri=model_uri,
    input_data=input_data,
    env_manager="local",
)
