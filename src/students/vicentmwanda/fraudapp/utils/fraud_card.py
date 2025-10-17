# scripts/utils.py
import os

import joblib
import numpy as np

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(MODEL_PATH)


def predict_fraud(features: list):
    """
    Predict credit card fraud using loaded model
    :param features: list of 30 features in correct order [Time, V1-V28, Amount]
    :return: Fraud prediction with probability
    """
    try:
        features_np = np.array(features).reshape(1, -1)
        prediction = model.predict(features_np)[0]
        prob = model.predict_proba(features_np)[0][1]
        level = prob * 100
        if prediction == 1:
            return f"div id='output_dt' data-val={level}>🚨 FRAUD DETECTED ({prob * 100:.2f}% probability)</div>"
        else:
            return f"<div id='output_dt' data-val={level}>✅ LEGITIMATE TRANSACTION ({(1 - prob) * 100:.2f}% probability)</div>"
    except Exception as e:
        return f"❌ Prediction Error: {str(e)}"


def predict_fraud_detailed(features: list):
    """
    Predict credit card fraud with detailed output
    :param features: list of 30 features in correct order [Time, V1-V28, Amount]
    :return: Dictionary with prediction details
    """
    try:
        features_np = np.array(features).reshape(1, -1)
        prediction = model.predict(features_np)[0]
        probabilities = model.predict_proba(features_np)[0]

        fraud_prob = probabilities[1] * 100
        legit_prob = probabilities[0] * 100

        result = {
            'prediction': prediction,
            'fraud_probability': fraud_prob,
            'legitimate_probability': legit_prob,
            'status': 'FRAUD' if prediction == 1 else 'LEGITIMATE',
            'risk_level': 'HIGH' if fraud_prob > 70 else 'MEDIUM' if fraud_prob > 30 else 'LOW'
        }

        return result
    except Exception as e:
        return {'error': str(e)}
