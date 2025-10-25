"""
Test script for the Fraud Detection model served by MLflow.
This script loads the model from MLflow Registry and makes predictions.
"""
import os

import mlflow
import numpy as np
import pandas as pd

MODEL_NAME = "fraud-detection-rf"
MODEL_STAGE = "Production"

# Construct an absolute path to the database file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "..", "..", "..", "..", "..", "mlflow_local_tracking.db")

# Construct the URI to load the model from the registry
model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

# Set up MLflow Tracking
mlflow.set_tracking_uri(f"sqlite:///{os.path.abspath(DB_PATH)}")

print("=" * 60)
print("FRAUD DETECTION MODEL TESTING")
print("=" * 60)
print(f"Model: {MODEL_NAME}")
print(f"Stage: {MODEL_STAGE}")
print(f"MLflow URI: {model_uri}")
print(f"Tracking DB: {os.path.abspath(DB_PATH)}")
print("=" * 60)

# Load the model
print("\n[1/3] Loading model from MLflow Registry...")
try:
    model = mlflow.pyfunc.load_model(model_uri)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Prepare test data
# The fraud detection model expects 30 features (V1-V28, Time, Amount)
print("\n[2/3] Preparing test data...")

# Create sample transactions (30 features)
# In real scenario, these would be actual transaction features
test_data = pd.DataFrame({
   'Time': [30000.0],
    'V1': [3.5], 'V2': [-2.8], 'V3': [4.2], 'V4': [-3.1], 'V5': [2.9],
    'V6': [-3.5], 'V7': [3.8], 'V8': [-2.5], 'V9': [4.1], 'V10': [-3.2],
    'V11': [3.3], 'V12': [-2.9], 'V13': [3.7], 'V14': [-3.4], 'V15': [2.8],
    'V16': [-3.6], 'V17': [3.9], 'V18': [-2.7], 'V19': [3.4], 'V20': [-3.1],
    'V21': [2.9], 'V22': [-3.3], 'V23': [3.5], 'V24': [-2.8], 'V25': [3.2],
    'V26': [-3.4], 'V27': [3.6], 'V28': [-2.9],
    'Amount': [5000.00]
})

print(f"✅ Created {len(test_data)} test transactions")
print("\nTest Data Preview:")
print(test_data.head())

# Make predictions
print("\n[3/3] Making predictions...")
try:
    predictions = model.predict(test_data)
    print("✅ Predictions completed!")

    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)

    for i, pred in enumerate(predictions):
        transaction_type = "🚨 FRAUD" if pred == 1 else "✅ NORMAL"
        amount = test_data.loc[i, 'Amount']
        print(f"Transaction {i + 1}: Amount=${amount:.2f} → {transaction_type} (Class: {pred})")

    # Summary statistics
    fraud_count = int(np.sum(predictions == 1))
    normal_count = int(np.sum(predictions == 0))
    fraud_rate = (fraud_count / len(predictions)) * 100

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total Transactions: {len(predictions)}")
    print(f"Fraudulent: {fraud_count} ({fraud_rate:.1f}%)")
    print(f"Normal: {normal_count} ({100 - fraud_rate:.1f}%)")
    print("=" * 60)

except Exception as e:
    print(f"❌ Error making predictions: {e}")
    exit(1)

print("\n✅ Model testing completed successfully!")
