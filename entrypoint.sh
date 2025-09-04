#!/bin/bash
set -e

if [ "$DEPLOYMENT_TARGET" = "STUDENT" ]; then
    # Assume you pass in the student identifier, e.g., student1.
    if [ -z "$GITHUB_USER" ]; then
        echo "Error: STUDENTID is not set for a student deployment."
        exit 1
    fi
    echo "Starting student API for $GITHUB_USER..."
    cd src/students/$GITHUB_USER
else
    echo "Starting root API..."
    cd src/students/example
fi

# Start dagster and mlflow (if present) from project root (/app)
echo "Launching /app/run_dagster.sh in background..."
/app/run_dagster.sh &
RUNDAGSTER_PID=$!
echo "run_dagster.sh pid=$RUNDAGSTER_PID"

# Start the FastAPI app in background on port 8000
echo "Starting FastAPI on port 8000..."
uv run uvicorn "api.main:app" --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!

echo "Starting nginx (will run in foreground)..."
# exec nginx in the foreground so it becomes PID 1 in the container
exec nginx -g 'daemon off;'
