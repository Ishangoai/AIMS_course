#!/bin/bash
set -e

if [ "$DEPLOYMENT_TARGET" = "STUDENT" ]; then
    # Assume you pass in the student identifier, e.g., student1.
    if [ -z "$GITHUB_USER" ]; then
        echo "Error: STUDENT_ID is not set for a student deployment."
        exit 1
    fi
    echo "Starting student API for $GITHUB_USER..."
    exec uv run uvicorn "students/${GITHUB_USER}/api/main:app" --host 0.0.0.0 --port 8080
else
    echo "Starting root API..."
    exec uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8080
fi
