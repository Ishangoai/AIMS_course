#!/usr/bin/env python3
"""
🎯 Local Test for Fraud Detection ML Model (MLflow)
"""

import json
import os

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

# ----------------------------------------
# ANSI Styling
# ----------------------------------------

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
GRAY = "\033[90m"

# ----------------------------------------
# Configuration
# ----------------------------------------

MODEL_NAME = "fraud_detection_rf"
MODEL_STAGE = "latest"
DB_PATH = os.path.join("mlflow_artifacts", "mlflow_fraud_tracking.db")
model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

# ----------------------------------------
# Setup MLflow Tracking
# ----------------------------------------

mlflow.set_tracking_uri(f"sqlite:///{os.path.abspath(DB_PATH)}")

print(BOLD + "=" * 60 + RESET)
print(BOLD + "🎯 Fraud Detection Model - Local Test" + RESET)
print(BOLD + "=" * 60 + RESET)

print(f"{CYAN}📂 Database:{RESET} {DB_PATH}")
print(f"{CYAN}📊 Model URI:{RESET} {model_uri}")
print(BOLD + "=" * 60 + RESET)

# ----------------------------------------
# Fetch input example from MLflow artifact
# ----------------------------------------

print(f"\n{YELLOW}🔍 Fetching input example from MLflow artifact...{RESET}")

client = MlflowClient()
model_versions = client.search_model_versions(f"name='{MODEL_NAME}'")

latest_version = model_versions[0]
run_id = latest_version.run_id
if run_id is None:
    raise ValueError("Failed to retrieve a valid run_id from the latest model version.")

print(f"{GREEN}✅ Found model version:{RESET} {latest_version.version}")
print(f"{GREEN}📦 Run ID:{RESET} {run_id}")

# Load the artifact containing input example
artifact_path = "random_forest_model"
local_path = client.download_artifacts(run_id, artifact_path)
input_example_path = os.path.join(local_path, "input_example.json")

with open(input_example_path, 'r') as f:
    example_data = json.load(f)

# Parse input data
if 'dataframe_split' in example_data:
    split = example_data['dataframe_split']
    input_data = pd.DataFrame(split['data'], columns=split['columns'])
elif 'data' in example_data and 'columns' in example_data:
    input_data = pd.DataFrame(example_data['data'], columns=example_data['columns'])
else:
    input_data = pd.DataFrame(example_data)

print(f"{GREEN}✅ Loaded input example from MLflow:{RESET} {input_data.shape}")
print(f"\n{BLUE}📋 Input shape:{RESET} {input_data.shape}")
print(f"{BLUE}📊 Features:{RESET} {list(input_data.columns)}")

# ----------------------------------------
# Features :
# ----------------------------------------
# Fetch Top 10 Important Features
# ----------------------------------------

print(f"\n{YELLOW}📥 Fetching top 10 important features...{RESET}")

top_features_path = os.path.join(local_path, "artifacts", "top_10_features.json")
with open(top_features_path, 'r') as f:
    top_features = json.load(f)

print(f"{GREEN}✅ Loaded top 10 features:{RESET} {top_features}")

# Filter input data to include only top 10 features
input_data = input_data[top_features]

# ----------------------------------------
# Run Predictions
# ----------------------------------------

print(f"\n{YELLOW}🔮 Running predictions...{RESET}")

loaded_model = mlflow.pyfunc.load_model(model_uri)
results = loaded_model.predict(input_data)

print("\n" + BOLD + "=" * 60 + RESET)
print(GREEN + BOLD + "✅ PREDICTIONS COMPLETE" + RESET)
print(BOLD + "=" * 60 + RESET)

# ----------------------------------------
# Display Results
# ----------------------------------------

# Print table header without Time and Amount
print(f"\n{BOLD}{'Transaction':<12} {'Prediction':<20}{RESET}")
print(f"{'-' * 32}")

# Print rows without Time and Amount
for i, pred in enumerate(results):
    label = f"{RED}🚨 FRAUD DETECTED{RESET}" if pred == 1 else f"{GREEN}✅ LEGITIMATE{RESET}"
    print(f"{i + 1:<12} {label:<20}")

# ----------------------------------------
# Summary
# ----------------------------------------

fraud_count = sum(results)
print("\n" + BOLD + "=" * 60 + RESET)
print(f"{CYAN}📊 SUMMARY:{RESET} {fraud_count}/{len(results)} transactions flagged as {RED}FRAUD{RESET}")
print(BOLD + "=" * 60 + RESET)
print(f"\n{GREEN}✅ Test completed successfully!{RESET}")
