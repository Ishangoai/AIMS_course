import io
import os
import textwrap

import gradio as gr
from api.models import UpdateUserRequest, UserRequest
from api.safe_eval import safe_eval
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from gradioapp import image_processor_app as processor
from PIL import Image, ImageEnhance

app = FastAPI(
    title="AIMS Course API",
    description=textwrap.dedent(
        """
        ## Mounted Apps
        ----
        1. [**General Gradio Demo**](/gradio/)
        2. [**Heart Disease Prediction App**](/heart-disease/)
        3. [**Simple LLM Chatbot**](/llm-chat/)
        -----
        """
    ),
    version="1.0.0",
    contact={"name": "Support Team", "email": "vincent@ishango.ai"},
    redirect_slashes=False,
)

# Global variable to store the usernames
current_user = os.environ.get("GITHUB_USER", "default")
users = {}


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
    """Evaluate the given arguments and return the result."""
    try:
        result = safe_eval(expression)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


@app.post(
    "/register",
    summary="Register a new user",
    description="Registers a new user with the given username.",
)
def register_user(request: UserRequest):
    """Register a new user with the given username."""
    username = request.username
    if not username:
        raise HTTPException(status_code=400, detail="Username is required")

    users[username] = request.model_dump().get("name", None)
    return {"message": f"User {username} registered successfully"}


@app.get("/register", summary="Get registered users", description="Returns a list of registered users.")
def get_registered_users():
    """Get the list of registered users."""
    return {"users": users}


@app.get("/register/{username}", summary="Get user details", description="Returns details of a specific user.")
def get_user_details(username: str):
    """Get the details of a specific user."""
    if username not in users:
        raise HTTPException(status_code=404, detail="User not found")
    return {"username": username, "name": users[username]}


@app.delete(
    "/register/{username}/delete",
    summary="Delete a user",
    description="Deletes a user with the given username.",
)
def delete_user(username: str):
    """Delete a user with the given username."""
    if username not in users:
        raise HTTPException(status_code=404, detail="User not found")
    del users[username]
    return {"message": f"User {username} deleted successfully"}


@app.put(
    "/register/{username}",
    summary="Update user details",
    description="Updates the details of a specific user.",
)
def update_user_details(username: str, request: UpdateUserRequest):
    """Update the details of a specific user."""
    if username not in users:
        raise HTTPException(status_code=404, detail="User not found")
    users[username] = request.model_dump().get("name", None)
    return {"message": f"User {username} updated successfully"}


###################################################################################################
# IMAGE PROCESSING ENDPOINTS
###################################################################################################


async def get_image_from_file(image_file):
    img = await image_file.read()
    img_bytes = io.BytesIO(img)
    return Image.open(img_bytes)


async def get_params_from_request(request: Request):
    form = await request.form()
    return {key: value for key, value in form.items() if key != "image"}


def convert_image_to_buffer(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


@app.post("/grayscale")
async def to_grayscale(image: UploadFile = File(...)):
    img = await get_image_from_file(image)
    gray_scale_img = img.convert("L")  # convert image into grayscale
    buffer = convert_image_to_buffer(gray_scale_img)
    return Response(content=buffer.getvalue(), media_type="image/png")


@app.post("/brightness")
async def adjust_brightness(request: Request, image: UploadFile = File(...)):
    params = await get_params_from_request(request)
    brightness = float(params["brightness"])
    img = await get_image_from_file(image)

    enhancer = ImageEnhance.Brightness(img)
    enhanced_img = enhancer.enhance(brightness)

    buffer = convert_image_to_buffer(enhanced_img)
    return Response(content=buffer.getvalue(), media_type="image/png")


@app.post("/contrast")
async def adjust_contrast(request: Request, image: UploadFile = File(...)):
    params = await get_params_from_request(request)
    contrast = float(params["contrast"])
    img = await get_image_from_file(image)

    enhancer = ImageEnhance.Contrast(img)
    enhanced_img = enhancer.enhance(contrast)

    buffer = convert_image_to_buffer(enhanced_img)
    return Response(content=buffer.getvalue(), media_type="image/png")


@app.post("/rotate")
async def rotate(request: Request, image: UploadFile = File(...)):
    params = await get_params_from_request(request)
    angle = float(params["rotation"])
    img = await get_image_from_file(image)

    rotated_img = img.rotate(angle, expand=True)
    buffer = convert_image_to_buffer(rotated_img)
    return Response(content=buffer.getvalue(), media_type="image/png")


# Mount Gradio app
demo = processor.create_app()
gr.mount_gradio_app(app, demo, path="/")  # Mount the targeted app at the root

# Used for later (not now)
# gr.mount_gradio_app(app, heart_app, path="/heart-disease")
# gr.mount_gradio_app(app, llm_chat, path="/llm-chat")
