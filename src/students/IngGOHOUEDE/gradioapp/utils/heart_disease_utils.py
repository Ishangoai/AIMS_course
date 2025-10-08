# scripts/utils.py
import os

import joblib
import numpy as np

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "heart_model.pkl")
model = joblib.load(MODEL_PATH)


def predict_heart_disease(features: list):
    """
    Predict heart disease using loaded model
    :param features: list of 13 features in correct order
    :return: "High Risk" or "Low risk"
    """
    try:
        features_np = np.array(features).reshape(1, -1)
        prediction = model.predict(features_np)[0]
        prob = model.predict_proba(features_np)[0][1]
        if prediction == 1:
            return f"High risk of heart disease ({prob * 100:.2f}% probability)"
        else:
            return f"Low risk of heart disease ({(1 - prob) * 100:.2f}% probability)"
    except Exception as e:
        return f"Error: {str(e)}"
