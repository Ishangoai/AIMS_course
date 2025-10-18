"""
Test script for the Fraud Detection model served by MLflow.
This script loads the model from MLflow Registry and makes predictions.
"""
import os
import mlflow
import pandas as pd
import numpy as np

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
    'Time': [0.0, 100.0, 200.0],
    'V1': [-1.5, 0.5, 2.0],
    'V2': [1.2, -0.8, 0.3],
    'V3': [0.5, 1.5, -1.0],
    'V4': [-0.3, 0.7, 1.2],
    'V5': [0.8, -0.5, 0.9],
    'V6': [-0.2, 1.0, -0.7],
    'V7': [0.4, -0.3, 0.6],
    'V8': [-0.6, 0.2, -0.4],
    'V9': [0.9, -1.2, 0.5],
    'V10': [-0.7, 0.6, -0.8],
    'V11': [0.3, -0.9, 1.1],
    'V12': [-1.0, 0.4, -0.5],
    'V13': [0.6, -0.6, 0.7],
    'V14': [-0.4, 0.8, -0.3],
    'V15': [0.2, -0.4, 0.4],
    'V16': [-0.8, 0.3, -0.6],
    'V17': [0.5, -0.7, 0.8],
    'V18': [-0.3, 0.5, -0.4],
    'V19': [0.7, -0.2, 0.3],
    'V20': [-0.5, 0.9, -0.7],
    'V21': [0.4, -0.3, 0.5],
    'V22': [-0.6, 0.6, -0.5],
    'V23': [0.8, -0.8, 0.6],
    'V24': [-0.2, 0.4, -0.3],
    'V25': [0.3, -0.5, 0.7],
    'V26': [-0.7, 0.2, -0.4],
    'V27': [0.5, -0.6, 0.2],
    'V28': [-0.4, 0.7, -0.6],
    'Amount': [10.50, 250.00, 1500.00]
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
        print(f"Transaction {i+1}: Amount=${amount:.2f} → {transaction_type} (Class: {pred})")
    
    # Summary statistics
    fraud_count = int(np.sum(predictions == 1))
    normal_count = int(np.sum(predictions == 0))
    fraud_rate = (fraud_count / len(predictions)) * 100
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total Transactions: {len(predictions)}")
    print(f"Fraudulent: {fraud_count} ({fraud_rate:.1f}%)")
    print(f"Normal: {normal_count} ({100-fraud_rate:.1f}%)")
    print("=" * 60)
    
except Exception as e:
    print(f"❌ Error making predictions: {e}")
    exit(1)

print("\n✅ Model testing completed successfully!")

