#!/bin/bash

# This script serves the production-ready ML model using the MLflow model server
# It is designed to be run from within the `ml_fraud` directory.

# Script Configuration
set -e # Stop the script if any command fails

# Model and environment details
MODEL_NAME="fraud_detection_model"
MODEL_STAGE="Production"
MLFLOW_PORT=5002

# Get the directory where this script is located (i.e., the ml_fraud folder)
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
echo "SCRIPT_DIR '$SCRIPT_DIR'"
# Construct the path to the database file, which is two directories up
# from the `ml_fraud` folder.
DB_PATH="$SCRIPT_DIR/../../../../../mlflow_local_tracking.db"

# Set the MLFLOW_TRACKING_URI for the mlflow command
export MLFLOW_TRACKING_URI="sqlite:///$DB_PATH"

echo "Starting the MLflow model server..."
echo "Model: '$MODEL_NAME' from stage '$MODEL_STAGE'"
echo "MLflow URI: $MLFLOW_TRACKING_URI"


# The --env-manager=local flag tells MLflow to use the current virtual environment
mlflow models serve -m "models:/$MODEL_NAME/$MODEL_STAGE" --port "$MLFLOW_PORT"  --env-manager=local