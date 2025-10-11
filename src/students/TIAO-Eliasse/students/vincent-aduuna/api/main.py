import os

from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html

from api.models import EvaluateRequest, UpdateUserRequest
from api.safe_eval import safe_eval

app = FastAPI(
    title="AIMS Course API",
    description="This API provides sample endpoints",
    version="1.0.0",
    contact={
        "name": "Support Team",
        "email": "vincent@ishango.ai",
    },
)

# Global variable to store the username
current_user = os.environ.get("GITHUB_USER", "default")


@app.get("/", include_in_schema=False)
def root():
    """
    Redirect the root path `/` to the Swagger UI documentation.
    """
    return get_swagger_ui_html(openapi_url="/openapi.json", title="AIMS Course API Docs")


@app.get("/hello", summary="Greet the user", description="Returns a greeting message.")
def hello():
    return {"message": f"Hello from {current_user}!"}


@app.put(
    "/user",
    summary="Update the username",
    description="Updates the username that is used in the /hello endpoint.",
    response_description="A confirmation message.",
)
def update_user(request: UpdateUserRequest):
    global current_user
    if not request.username:
        raise HTTPException(status_code=400, detail="Username cannot be empty")
    current_user = request.username
    return {"message": f"Username updated to {current_user}"}


@app.post(
    "/evaluate",
    summary="Evaluate an expression",
    description="Evaluates a Math expression provided in the request body.",
    response_description="The result of the evaluated expression.",
)
def evaluate(args: EvaluateRequest):
    """
    Evaluate the given arguments and return the result.
    """
    try:
        # Evaluate the expression
        print(args.expression)
        print(type(args.expression))
        result = safe_eval(args.expression)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}
