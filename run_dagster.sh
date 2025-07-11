#!/bin/bash


echo "MLflow UI starting on port 5000..."
mlflow ui --backend-store-uri sqlite:///mlflow_local_tracking.db &
MLFLOW_PID=$!

# Ensure MLflow UI is stopped when the script exits
trap "kill $MLFLOW_PID 2>/dev/null" EXIT

if [[ "$GITHUB_USER" == "oliverangelil" || "$GITHUB_USER" == "aduuna" || "$GITHUB_USER" == "cyrille-feu" ]]; then
    echo "Running Dagster for user: example"
    dagster dev -m "src.students.example.mlflow.era5_temperature_project.era5_pipeline.definitions"
else
    echo "Attempting to run Dagster for user: $GITHUB_USER"
    dagster dev -m "src.students.$GITHUB_USER.mlflow.era5_temperature_project.era5_pipeline.definitions"
fi
