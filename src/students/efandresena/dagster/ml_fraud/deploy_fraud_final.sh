#!/bin/bash
# This script serves the fraud detection ML model using the MLflow model server

# Script Configuration
set -e # Stop the script if any command fails

# Model and environment details
MODEL_NAME="fraud_detection_rf"
MODEL_STAGE="latest"
MLFLOW_PORT=5001

# Get the directory where this script is located
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
echo "$SCRIPT_DIR"
# Construct the path to the database (3 levels up from ml_fraud/)
DB_PATH="$SCRIPT_DIR/mlflow_artifacts/mlflow_fraud_tracking.db"

# Check if database exists
if [ ! -f "$DB_PATH" ]; then
    echo "❌ Error: Database not found at $DB_PATH"
    exit 1
fi

# Set the MLFLOW_TRACKING_URI for the mlflow command
export MLFLOW_TRACKING_URI="sqlite:///$DB_PATH"

echo "=========================================="
echo "🚀 Starting Fraud Detection Model Server"
echo "=========================================="
echo "📊 Model: '$MODEL_NAME' from stage '$MODEL_STAGE'"
echo "🌐 Port: $MLFLOW_PORT"
echo "💾 Database: $DB_PATH"
echo "🔗 URL: http://127.0.0.1:$MLFLOW_PORT"
echo "=========================================="
echo ""
echo "Server starting... (Press Ctrl+C to stop)"
echo ""

# The --env-manager=local flag tells MLflow to use the current virtual environment
mlflow models serve -m "models:/$MODEL_NAME/$MODEL_STAGE" --port "$MLFLOW_PORT" --env-manager=local
