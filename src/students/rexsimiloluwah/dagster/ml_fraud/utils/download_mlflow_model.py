import os

import joblib
import mlflow

MODEL_NAME = "fraud-detection-model"
MODEL_STAGE = "Production"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "..", "..", "..", "..", "..", "mlflow_local_tracking.db")

mlflow.set_tracking_uri(DB_PATH)

# Load the model
model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
loaded_model = mlflow.pyfunc.load_model(model_uri)

# Extract the underlying sklearn model (if it’s a sklearn flavor)
try:
    sklearn_model = loaded_model._model_impl  # if logged as pyfunc
except AttributeError:
    sklearn_model = loaded_model  # if directly loaded

# Save as pickle for reuse in Gradio app
joblib.dump(sklearn_model, "fraud_model.pkl")
print("Saved model to fraud_model.pkl ✅")
