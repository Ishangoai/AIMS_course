import mlflow
import pandas as pd
import numpy as np

# Set the tracking URI if needed
mlflow.set_tracking_uri("sqlite:///./mlruns/mlflow_local_tracking.db")  # or your MLflow URI

def load_last_promoted_model(model_name="RandomForestFraudModel"):
    """
    Load the last promoted model from MLflow registry.
    """
    try:
        # Search for the latest "Production" version
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(name=model_name, stages=["Production"])
        if not versions:
            raise ValueError("No model in Production stage found")
        model_uri = f"models:/{model_name}/Production"
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

FEATURE_NAMES = [f"V{i}" for i in range(1, 29)] + ["Amount"]

def predict_fraud(features: list):
    model = load_last_promoted_model()
    if model is None:
        return "No model loaded"
    
    # Convert input to DataFrame with proper column names
    df = pd.DataFrame([features], columns=FEATURE_NAMES)
    
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    return f"Prediction: {prediction}, Probability of Fraud: {probability:.2f}"
