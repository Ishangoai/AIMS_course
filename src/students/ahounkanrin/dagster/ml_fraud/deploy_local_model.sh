#!/bin/bash

# This script serves the production-ready ML model using the MLflow model server

# Script Configuration
set -e # Stop the script if any command fails

# Model and environment details
MODEL_NAME="tuned-fraud-detector"
MODEL_STAGE="Staging"
MLFLOW_PORT=5002

# Construct the path to the database
DB_PATH="/workspaces/AIMS_course/mlflow_local_tracking.db"

# Set the MLFLOW_TRACKING_URI for the mlflow command
export MLFLOW_TRACKING_URI="sqlite:///$DB_PATH"

# The --env-manager=local flag tells MLflow to use the current virtual environment
mlflow models serve -m "models:/$MODEL_NAME/$MODEL_STAGE" --port "$MLFLOW_PORT"  --env-manager=local