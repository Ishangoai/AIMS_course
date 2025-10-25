# scripts/utils.py
import os

import joblib

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "fraud_detection_model.pkl")
model = joblib.load(MODEL_PATH)
