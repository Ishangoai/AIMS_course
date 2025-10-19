#!/bin/bash

# Test script for MLflow model server
# This sends a sample transaction to the fraud detection model

set -e

# Configuration
MLFLOW_PORT=5002
ENDPOINT="http://localhost:$MLFLOW_PORT/invocations"

echo "Testing MLflow model server at $ENDPOINT"
echo "----------------------------------------"

# Sample data - Normal transaction (first sample)
NORMAL_TRANSACTION='{
  "dataframe_split": {
    "columns": ["V14", "V17", "V10", "V12", "V11", "V16", "V4", "V9", "V18", "V7", "V3", "Amount"],
    "data": [[0.35237478, -0.010299228, -0.156840906, 0.538299087, 0.847755891, 0.906489408, -0.146817811, -0.49952308, 0.576837916, -0.166862547, 0.220932254, 8.99]]
  }
}'

# Sample data - Fraud transaction (second sample)
FRAUD_TRANSACTION='{
  "dataframe_split": {
    "columns": ["V14", "V17", "V10", "V12", "V11", "V16", "V4", "V9", "V18", "V7", "V3", "Amount"],
    "data": [[-10.5, -15.0, -5.0, -7.5, 5.0, -8.0, 5.0, -3.0, -4.0, -1.0, -2.0, 500.0]]
  }
}'

# Test normal transaction
echo "Testing NORMAL transaction..."
curl -X POST "$ENDPOINT" \
  -H "Content-Type: application/json" \
  -d "$NORMAL_TRANSACTION" \
  | jq '.'

echo ""
echo "----------------------------------------"

# Test fraud transaction
echo "Testing FRAUD transaction..."
curl -X POST "$ENDPOINT" \
  -H "Content-Type: application/json" \
  -d "$FRAUD_TRANSACTION" \
  | jq '.'

echo ""
echo "Testing complete!"