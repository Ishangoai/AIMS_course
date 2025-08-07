import os

import gradio as gr
from agents.ai_agent.llm_gradio import llm_chat as agentic_llm_chat
from agents.chatbot.llm_gradio import llm_chat
from api.models import UpdateUserRequest, UserRequest
from api.safe_eval import safe_eval
from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from gradioapp.app import app as demo
from gradioapp.heart_disease_app import heart_app

app = FastAPI(
    title="AIMS Course API",
    description="""
    This API provides sample endpoints
    """,
    version="1.0.0",
    contact={
        "name": "Support Team",
        "email": "vincent@ishango.ai",
    },
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


def custom_openapi():
    # cif already generated, return the cached schema
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        contact=app.contact,
        routes=app.routes,
    )

    html_response = {
        "200": {
            "description": "The HTML page for the Gradio application.",
            "content": {"text/html": {"schema": {"type": "string"}}},
        }
    }

    openapi_schema["paths"]["/gradio"] = {
        "get": {
            "summary": "Gradio Demo UI",
            "description": "Opens the interactive Gradio demo application.",
            "tags": ["Mounted Gradio Apps"],
            "responses": html_response,
        }
    }
    openapi_schema["paths"]["/heart-disease"] = {
        "get": {
            "summary": "Heart Disease Prediction App",
            "description": "Opens an app to predict heart disease based on patient data",
            "tags": ["Mounted Gradio Apps"],
            "responses": html_response,
        }
    }
    openapi_schema["paths"]["/llm-chat"] = {
        "get": {
            "summary": "Simple LLM Chatbot",
            "description": "Opens a simple chat interface with an LLM",
            "tags": ["Mounted Gradio Apps"],
            "responses": html_response,
        }
    }
    openapi_schema["paths"]["/agentic-llm-chat"] = {
        "get": {
            "summary": "Agentic LLM Chatbot",
            "description": "Opens an advanced, agentic chat interface with an LLM",
            "tags": ["Mounted Gradio Apps"],
            "responses": html_response,
        }
    }

    # Cache and return the modified schema
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


gr.mount_gradio_app(app, demo, path="/gradio")
gr.mount_gradio_app(app, heart_app, path="/heart-disease")
gr.mount_gradio_app(app, llm_chat, path="/llm-chat")
gr.mount_gradio_app(app, agentic_llm_chat, path="/agentic-llm-chat")
