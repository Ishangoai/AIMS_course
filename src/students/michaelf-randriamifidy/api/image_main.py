import os
import textwrap

import gradio as gr

from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html 
from gradioapp.image_app import image_app

app = FastAPI(
    title="AIMS Course API",
    description=textwrap.dedent("""
    ----
    1. [**Image Transformation**](/image-transformation/)
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
    return {"message": f"Hello from {current_user}! Ready!!"}

gr.mount_gradio_app(app, image_app, path="/image-transformation")
