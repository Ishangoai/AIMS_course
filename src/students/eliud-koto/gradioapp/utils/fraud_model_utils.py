# scripts/utils.py
import os

import joblib
import mlflow.sklearn
import numpy as np

mlflow.set_tracking_uri("http://localhost:5000")


# base_dir = "/var/autofs/misc/home/eliud/Desktop/AIMS_course"
# model_path = os.path.join(
#     base_dir,
#     "mlruns",
#     "1",
#     "562ddd67cc6b49619f8e7aea68cd6cfd",
#     "artifacts",
#     "tuned_random_forest_v1760881924.pkl"
# )

# model = joblib.load(model_path)

# # # Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "fraud_model.pkl")
model = joblib.load(MODEL_PATH)


def predict_fraud_activity(features: list):
    """
    Predict fraud activity using loaded model
    :param features: list of 31 features in correct order
    :return: "High Fruad" or "Low Fruad"
    """
    try:
        features_np = np.array(features).reshape(1, -1)
        prediction = model.predict(features_np)[0]
        prob = model.predict_proba(features_np)[0][1]
        if prediction == 0:
            return f"Low risk of fraud ({prob * 100:.2f}% probability)"
        else:
            return f"Low risk of fraud({(1 - prob) * 100:.2f}% probability)"
    except Exception as e:
        return f"Error: {str(e)}"
