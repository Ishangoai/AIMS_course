# fraud_utils.py
import os

import joblib
import pandas as pd

# Use your exact model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "fraud_model.pkl")

# Load once at import
model = joblib.load(MODEL_PATH)

# Feature names as expected by model
FEATURE_NAMES = [f"V{i}" for i in range(1, 29)] + ["Amount"]


def predict_fraud(features: list):  # type:ignore
    try:
        df = pd.DataFrame([features], columns=FEATURE_NAMES)  # type:ignore

        # Fix column name mismatch
        if "Amount" in df.columns:
            df["Amount_scaled"] = df["Amount"]
            df = df.drop(columns=["Amount"])

        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        if prediction == 1:
            return f"Fraudulent transaction detected ({probability * 100:.2f}% probability)"
        else:
            return f"Legitimate transaction ({(1 - probability) * 100:.2f}% probability)"
    except Exception as e:
        return f"Error during prediction: {str(e)}"
