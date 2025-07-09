#!/bin/bash

if [[ "$GITHUB_USER" == "oliverangelil" || "$GITHUB_USER" == "aduuna" || "$GITHUB_USER" == "cyrille-feu" ]]; then
    echo "Running Dagster for user: example"
    dagster dev -m "src.students.example.mlflow.era5_temperature_project.era5_pipeline.definitions"
else
    echo "Attempting to run Dagster for user: $GITHUB_USER"
    dagster dev -m "src.students.$GITHUB_USER.mlflow.era5_temperature_project.era5_pipeline.definitions"
fi
