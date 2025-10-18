# scripts/utils.py
import os

import joblib
import numpy as np

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "tuned_random_forest.pkl")
model = joblib.load(MODEL_PATH)


def predict_fraud(features: list):
    """
    Predict fraud using loaded model
    :param features: list of 30 features in correct order
    :return: "Fraud" or "Not fraud"
    """
    try:
        features_np = np.array(features).reshape(1, -1)
        prediction = model.predict(features_np)[0]
        prob = model.predict_proba(features_np)[0][1]
        if prediction == 1:
            return f"Fraud detected with ({prob * 100:.2f}% probability)"
        else:
            return f"No fraud detected with ({(1 - prob) * 100:.2f}% probability)"
    except Exception as e:
        return f"Error: {str(e)}"
