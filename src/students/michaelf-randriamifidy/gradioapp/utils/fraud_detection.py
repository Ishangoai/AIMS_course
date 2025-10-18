import os

import joblib
import numpy as np
import pandas as pd


def is_valid_csv_file(file_path: str) -> bool:
    try:
        pd.read_csv(file_path)

        return True
    except Exception:
        return False


def predict_fraud(features_file: str) -> np.ndarray | None:
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "fraud_detector.pkl")

    try:
        # Load model
        model = joblib.load(MODEL_PATH)

        # Load features
        X = pd.read_csv(features_file)

        # Make predictions
        predictions = model.predict(X)

        return predictions
    except Exception as e:
        print(f"Error details: {e}")
        return None
