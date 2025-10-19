#!/bin/bash

# This script serves the production-ready Fraud Detection ML model using the MLflow model server

# Script Configuration
set -e # Stop the script if any command fails

# Model and environment details
MODEL_NAME="fraud-detection-rf"
MODEL_STAGE="Production"
MLFLOW_PORT=5003

# Get the directory where this script is located
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# Construct the path to the database
DB_PATH="$SCRIPT_DIR/../../../../../mlflow_local_tracking.db"

# Set the MLFLOW_TRACKING_URI for the mlflow command
export MLFLOW_TRACKING_URI="sqlite:///$DB_PATH"

echo "=========================================="
echo "Starting MLflow Fraud Detection Model Server"
echo "=========================================="
echo "Model: '$MODEL_NAME' from stage '$MODEL_STAGE'"
echo "Port: $MLFLOW_PORT"
echo "Tracking URI: $MLFLOW_TRACKING_URI"
echo "=========================================="

# The --env-manager=local flag tells MLflow to use the current virtual environment
mlflow models serve -m "models:/$MODEL_NAME/$MODEL_STAGE" --port "$MLFLOW_PORT" --env-manager=local