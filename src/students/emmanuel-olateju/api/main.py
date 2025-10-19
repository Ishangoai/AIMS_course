import os
import textwrap

import gradio as gr
from agents.chatbot.llm_gradio import llm_chat
from api.models import UpdateUserRequest, UserRequest
from api.safe_eval import safe_eval
from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from gradioapp.app import app as demo
from gradioapp.fraud_detection import fd_app
from gradioapp.heart_disease_app import heart_app
from gradioapp.imagedit import image_edit_app
from gradioapp.imagedit_v2 import image_edit_v2_app
from gradioapp.imagedit_vibe import imagedit_vibe_app

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
    5. [**Fraud Detection Model Endpoint**](/fraud-detection/invocations) - POST endpoint
    6. [**Fraud Detection App**](/fraud-detection) - POST endpoint
    -----
    """),
    version="1.0.0",
    contact={"name": "Support Team", "email": "vincent@ishango.ai"},
    redirect_slashes=False,
)

# Global variable to store the usernames
current_user = os.environ.get("GITHUB_USER", "default")
users = {}


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
gr.mount_gradio_app(app, fd_app, path="/fraud-detection")
# gr.mount_gradio_app(app, image_editor, path="/image-edit-v3")
