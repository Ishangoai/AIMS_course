import os
from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html

from api.models import UserRequest
from api.safe_eval import safe_eval
import gradio as gr
# from gradio.routes import App as GradioApp
# from gradioapp.app import demo

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

# Global variable to store the username
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


# @app.put(
#     "/user",
#     summary="Update the username",
#     description="Updates the username that is used in the /hello endpoint.",
#     response_description="A confirmation message.",
# )
# def update_user(request: UserRequest):
#     global current_user
#     if not request.username:
#         raise HTTPException(status_code=400, detail="Username cannot be empty")
#     current_user = request.username
#     return {"message": f"Username updated to {current_user}"}


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

    users[username] = request.model_dump()
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
    return {"username": username, "details": users[username]}


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
def update_user_details(username: str, request: UserRequest):
    """
    Update the details of a specific user.
    """
    # Here you would typically update the user in a database
    if username not in users:
        raise HTTPException(status_code=404, detail="User not found")
    del users[username]
    users[request.username] = request.model_dump()
    return {"message": f"User {username} updated successfully"}


demo = gr.Interface(fn=lambda x: "hello " + x, inputs="text", outputs="text")
gr.mount_gradio_app(app, demo, path="/gradio")
