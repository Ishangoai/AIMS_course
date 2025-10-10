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

exec uv run uvicorn "api.main:app" --host 0.0.0.0 --port 8080