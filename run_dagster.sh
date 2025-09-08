#!/bin/bash


echo "MLflow UI starting on port 5000..."
uv run mlflow ui --backend-store-uri sqlite:///mlflow_local_tracking.db &
MLFLOW_PID=$!

# Ensure MLflow UI is stopped when the script exits
trap "kill $MLFLOW_PID 2>/dev/null" EXIT

if [[ "$GITHUB_USER" == "oliverangelil" || "$GITHUB_USER" == "aduuna" || "$GITHUB_USER" == "cyrille-feu" ]]; then
    echo "Running Dagster for user: example"
    uv run dagster dev -m "src.students.example.dagster.definitions"
else
    echo "Attempting to run Dagster for user: $GITHUB_USER"
    uv run dagster dev -m "src.students.$GITHUB_USER.dagster.definitions"
fi
