import os
import textwrap
from typing import List, Optional

import gradio as gr
import mlflow
import numpy as np
from agents.chatbot.llm_gradio import llm_chat
from api.models import UpdateUserRequest, UserRequest
from api.safe_eval import safe_eval
from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from gradioapp.app import app as demo
from gradioapp.heart_disease_app import heart_app
from gradioapp.imagedit import image_edit_app
from gradioapp.imagedit_v2 import image_edit_v2_app
from gradioapp.imagedit_vibe import imagedit_vibe_app
from pydantic import BaseModel, Field

# from gradioapp.imagedit_v3 import image_editor

app = FastAPI(
    title="AIMS Course API",
    description=textwrap.dedent("""
    ## Mounted Apps
    ----
    1. [**General Gradio Demo**](/gradio/)
    2. [**Heart Disease Prediction App**](/heart-disease/)
    3. [**Simple LLM Chatbot**](/llm-chat/)
    4. [**Imagedit 🖼️](/image-edit/)
    5. [**Fraud Detection Model**](/fraud-detection/invocations) - POST endpoint
    -----
    """),
    version="1.0.0",
    contact={"name": "Support Team", "email": "vincent@ishango.ai"},
    redirect_slashes=False,
)

# Global variable to store the usernames
current_user = os.environ.get("GITHUB_USER", "default")
users = {}

# MLflow Configuration
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "fraud_detection_experiment"

# Global variable to store the loaded model
loaded_model = None


def load_fraud_detection_model():
    """
    Load the best model from MLflow.
    This can load either by run_id or by using the latest model from the experiment.
    """
    global loaded_model

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        # Option 1: Load a specific model by run_id
        # Uncomment and set your run_id if you want to load a specific run
        # run_id = "your_run_id_here"
        # loaded_model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

        # Option 2: Load the latest model from the experiment (recommended)
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

        if experiment is None:
            print(f"Warning: Experiment '{EXPERIMENT_NAME}' not found. Model will not be loaded.")
            return None

        # Get all runs from the experiment, sorted by metrics
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.roc_auc DESC"],  # Sort by ROC-AUC, you can change this
            max_results=1
        )

        if not runs:
            print(f"Warning: No runs found in experiment '{EXPERIMENT_NAME}'")
            return None

        best_run = runs[0]
        run_id = best_run.info.run_id

        # Load the model
        model_uri = f"runs:/{run_id}/model"
        loaded_model = mlflow.sklearn.load_model(model_uri)

        print(f"✅ Model loaded successfully from run: {run_id}")
        print(f"   ROC-AUC: {best_run.data.metrics.get('roc_auc', 'N/A')}")
        print(f"   Accuracy: {best_run.data.metrics.get('accuracy', 'N/A')}")

        return loaded_model

    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return None


async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    """
    # Startup: Load the model
    print("🚀 Starting up FastAPI application...")
    load_fraud_detection_model()
    if loaded_model is None:
        print("⚠️  Warning: Running without a trained model. Predictions will be random.")

    yield

    # Shutdown: Cleanup if needed
    print("👋 Shutting down FastAPI application...")


# Global variable to store the usernames
current_user = os.environ.get("GITHUB_USER", "default")
users = {}


# Pydantic models for fraud detection
class FraudPredictionRequest(BaseModel):
    """Request model for fraud predictions"""
    inputs: Optional[List[List[float]]] = Field(None, description="Input data as list of lists")
    data: Optional[List[List[float]]] = Field(None, description="Alternative field name for input data")

    class Config:
        json_schema_extra = {
            "example": {
                "inputs": [[62800.0, -1.508, 1.853, 0.220, -0.146, -0.528, -0.626, -0.166, 
                           0.999, -0.499, -0.156, 0.847, 0.538, -0.354, 0.352, 0.234, 
                           0.906, -0.010, 0.576, 0.256, 0.130, -0.218, -0.752, 0.059, 
                           -0.071, -0.017, 0.090, 0.204, 0.075, 8.99]]
            }
        }


class FraudPredictionResponse(BaseModel):
    """Response model for fraud predictions"""
    predictions: List[str] = Field(..., description="List of predictions (Fraud/Safe)")


@app.get("/", include_in_schema=False)
def root():
    """
    Redirect the root path `/` to the Swagger UI documentation.
    """
    return get_swagger_ui_html(openapi_url="/openapi.json", title="AIMS Course API Docs")


@app.get("/hello", summary="Greet the user", description="Returns a greeting message.")
def hello():
    return {"message": f"Hello from {current_user}!"}


@app.get(
    "/evaluate",
    summary="Evaluate an expression",
    description="Evaluates a Math expression provided in the request body.",
    response_description="The result of the evaluated expression.",
)
def evaluate(expression: str):
    """
    Evaluate the given arguments and return the result.
    """
    try:
        # Evaluate the expression
        # print(args.expression)
        # print(type(args.expression))
        result = safe_eval(expression)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


@app.post("/register", summary="Register a new user", description="Registers a new user with the given username.")
def register_user(request: UserRequest):
    """
    Register a new user with the given username.
    """
    # Extract the username from the request
    username = request.username
    if not username:
        raise HTTPException(status_code=400, detail="Username is required")

    users[username] = request.model_dump().get("name", None)
    return {"message": f"User {username} registered successfully"}


@app.get("/register", summary="Get registered users", description="Returns a list of registered users.")
def get_registered_users():
    """
    Get the list of registered users.
    """
    # Here you would typically fetch the users from a database
    return {"users": users}


@app.get("/register/{username}", summary="Get user details", description="Returns details of a specific user.")
def get_user_details(username: str):
    """
    Get the details of a specific user.
    """
    # Here you would typically fetch the user from a database
    if username not in users:
        raise HTTPException(status_code=404, detail="User not found")
    return {"username": username, "name": users[username]}


@app.delete(
    "/register/{username}/delete",
    summary="Delete a user",
    description="Deletes a user with the given username."
)
def delete_user(username: str):
    """
    Delete a user with the given username.
    """
    # Here you would typically delete the user from a database
    if username not in users:
        raise HTTPException(status_code=404, detail="User not found")
    del users[username]
    return {"message": f"User {username} deleted successfully"}


@app.put("/register/{username}", summary="Update user details", description="Updates the details of a specific user.")
def update_user_details(username: str, request: UpdateUserRequest):
    """
    Update the details of a specific user.
    """
    # Here you would typically update the user in a database
    if username not in users:
        raise HTTPException(status_code=404, detail="User not found")
    users[username] = request.model_dump().get("name", None)
    return {"message": f"User {username} updated successfully"}


# Fraud Detection Endpoints
@app.post(
    "/fraud-detection/invocations",
    response_model=FraudPredictionResponse,
    summary="Fraud detection predictions",
    description="Submit transaction data to get fraud predictions from the model",
    response_description="Predictions for each input sample"
)
async def fraud_detection_predict(request: FraudPredictionRequest):
    """
    Model inference endpoint for fraud detection.
    Accepts JSON with input data and returns fraud predictions.
    """
    try:
        # Extract inputs from request
        inputs = request.inputs if request.inputs is not None else request.data

        if inputs is None:
            raise HTTPException(
                status_code=400, 
                detail="Either 'inputs' or 'data' field is required"
            )

        # Convert to numpy array
        inputs_array = np.array(inputs)

        # Check if model is loaded
        if loaded_model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please ensure MLflow server is running and model is trained."
            )

        # Make predictions using the loaded model
        predictions_numeric = loaded_model.predict(inputs_array)

        # Convert numeric predictions to labels
        predictions = ["Fraud" if pred == 1 else "Safe" for pred in predictions_numeric]

        # Dummy prediction logic - replace with your actual model
        # For now, just return random predictions based on number of samples
        # num_samples = len(inputs_array) if len(inputs_array.shape) > 1 else 1
        # predictions = [
        #     "Fraud" if np.random.rand() > 0.5 else "Safe" 
        #     for _ in range(num_samples)
        # ]

        return FraudPredictionResponse(predictions=predictions)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get(
    "/fraud-detection/health",
    summary="Fraud detection health check",
    description="Check if the fraud detection service is running properly"
)
async def fraud_detection_health():
    """Health check endpoint for fraud detection service."""
    return {"status": "healthy"}


@app.get(
    "/fraud-detection/ping",
    summary="Fraud detection ping",
    description="MLflow compatibility ping endpoint"
)
async def fraud_detection_ping():
    """Ping endpoint for MLflow compatibility."""
    return {"status": "ok"}


gr.mount_gradio_app(app, demo, path="/gradio")
gr.mount_gradio_app(app, heart_app, path="/heart-disease")
gr.mount_gradio_app(app, llm_chat, path="/llm-chat")
gr.mount_gradio_app(app, image_edit_app, path="/image-edit")
gr.mount_gradio_app(app, image_edit_v2_app, path="/image-edit-v2")
gr.mount_gradio_app(app, imagedit_vibe_app, path="/image-edit-vibe")
# gr.mount_gradio_app(app, image_editor, path="/image-edit-v3")
