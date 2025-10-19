#!/bin/bash

# This script deploys the Gradio fraud detection app

# Script Configuration
set -e # Stop the script if any command fails

# Get the directory where this script is located
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# Change to the script directory
cd "$SCRIPT_DIR"

echo "🚀 Deploying Credit Card Fraud Detection Gradio App"
echo "📁 Working directory: $SCRIPT_DIR"

# Check if gradio is installed
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed or not in PATH"
    exit 1
fi

# Check if the gradio app file exists
if [ ! -f "gradio_app.py" ]; then
    echo "❌ gradio_app.py not found in current directory"
    exit 1
fi

# Check if the model server is running on port 5001
if ! nc -z localhost 5001; then
    echo "⚠️  Connection to model server on port 5001 failed."
    echo "   Please ensure the model server is running before starting the Gradio app."
    echo "   You can start it by running: ./deploy_local_model.sh"
    echo "   -----------------------------------------------------"
fi

# Check if MLflow database exists
DB_PATH="$SCRIPT_DIR/../../../../../mlflow_local_tracking.db"
if [ ! -f "$DB_PATH" ]; then
    echo "⚠️  MLflow database not found at: $DB_PATH"
    echo "   Make sure you have run the Dagster pipeline to train and register the model"
    echo "   The app will still launch but may not load the model correctly"
fi

echo "🔧 Starting Gradio application client..."
echo "🌐 App will be available at: http://localhost:7860"
echo "📡 Connecting to model server at http://localhost:5001"
echo ""
echo "Press Ctrl+C to stop the application"

# Launch the Gradio app
python gradio_app.py
