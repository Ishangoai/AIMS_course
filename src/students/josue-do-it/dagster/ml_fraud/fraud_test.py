import os

import mlflow
import numpy as np
import pandas as pd

# --- Configuration ---
MODEL_NAME = "fraud-detection-model"
MODEL_STAGE = "Production"
# MLFLOW_PORT is for the server, not used by this client script when using file-based registry
MLFLOW_PORT = 5002

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "..", "..", "..", "..", "..", "mlflow_local_tracking.db")

model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

# --- Set up MLflow Tracking ---
try:
    mlflow.set_tracking_uri(f"sqlite:///{os.path.abspath(DB_PATH)}")
    print(f"MLflow Tracking URI set to: sqlite:///{os.path.abspath(DB_PATH)}")
except Exception as e:
    print(f"Warning: Could not set MLflow tracking URI. Ensure the path is correct. Error: {e}")

# --- Prepare Input Data ---
# The input DataFrame MUST contain only the 12 features that the model was trained on,
# and the column order should ideally match the training data.
SELECTED_FEATURES = [
    "V14", "V17", "V10", "V12", "V11", "V16", "V4", "V9", "V18", "V7", "V3", "Amount"
]

sample_data = {
    "V14": [0.35237478, -10.5],
    "V17": [-0.010299228, -15.0],
    "V10": [-0.156840906, -5.0],
    "V12": [0.538299087, -7.5],
    "V11": [0.847755891, 5.0],
    "V16": [0.906489408, -8.0],
    "V4": [-0.146817811, 5.0],
    "V9": [-0.49952308, -3.0],
    "V18": [0.576837916, -4.0],
    "V7": [-0.166862547, -1.0],
    "V3": [0.220932254, -2.0],
    "Amount": [8.99, 500.0],
}

# Create DataFrame, ensuring columns are strictly in the required order
input_data = pd.DataFrame(sample_data, columns=SELECTED_FEATURES)  # type: ignore[call-overload]

print("\n--- Input Data for Prediction (showing 2 examples) ---")
print(input_data)


# --- Run Prediction ---
print(f"\nLoading and predicting with model: {model_uri}...")

try:
    # 1. Explicitly load the model first using pyfunc to ensure stable access
    loaded_model = mlflow.pyfunc.load_model(model_uri)

    # 2. Call the predict method on the loaded model object directly
    results = loaded_model.predict(
        data=input_data
    )

    # Explicitly check for valid results
    if results is None:
        raise ValueError("Model prediction returned None results unexpectedly.")

    if not isinstance(results, (np.ndarray, list, pd.Series)) or len(results) == 0:
        raise ValueError("Model prediction returned an empty or unrecognized result format.")

    print("\n--- Prediction Results (0: Non-Fraud, 1: Fraud) ---")

    predictions = results.tolist() if hasattr(results, 'tolist') else list(results)  # type: ignore

    prediction_df = input_data.copy()
    prediction_df['Prediction'] = predictions

    print(prediction_df)

except Exception as e:
    print("\n--- FATAL ERROR ---")

    if "No model with name" in str(e) or "RestException" in type(e).__name__:
        print(f"MLflow Model Loading Failed: Could not find or load model '{model_uri}'.")
        print(
            "ACTION REQUIRED: Check that the model name ('fraud-detection-model')",
            "and stage ('Production') are correct."
        )
        print("Also ensure the MLflow Tracking URI is set correctly and the database file exists.")
    else:
        print(f"An unexpected error occurred during prediction: {e}")
