from __future__ import annotations

import io
import textwrap
from typing import Any, Dict

import gradio as gr
from fastapi import FastAPI, File, Request, Response, UploadFile
from fastapi.openapi.docs import get_swagger_ui_html
from gradioapp import image_processor_app as processor  # your Gradio UI
from PIL import Image, ImageEnhance

# -----------------------------------------------------------------------------
# FastAPI app definition
# -----------------------------------------------------------------------------
app = FastAPI(
    title="AIMS Course API",
    description=textwrap.dedent("""
    ## Mounted Apps
    ----
    1. [**General Gradio Demo**](/gradio/)
    2. [**Heart Disease Prediction App**](/heart-disease/)
    3. [**Simple LLM Chatbot**](/llm-chat/)
    4. [**Simple Image editor**](/image-editor/)
    -----
    """),
    version="1.0.0",
    contact={"name": "Support Team", "email": "vincent@ishango.ai"},
    redirect_slashes=False,
)


@app.get("/", include_in_schema=False)
def root():
    """
    Redirect the root path `/` to the Swagger UI documentation.
    """
    return get_swagger_ui_html(openapi_url="/openapi.json", title="AIMS Course API Docs")


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
async def get_image_from_file(image_file: UploadFile) -> Image.Image:
    """Read uploaded image into a Pillow Image object."""
    data = await image_file.read()
    return Image.open(io.BytesIO(data))


async def get_params_from_request(request: Request) -> Dict[str, Any]:
    """Extract form parameters except the image itself."""
    form = await request.form()
    return {key: value for key, value in form.items() if key != "image"}


def image_to_response(image: Image.Image) -> Response:
    """Convert a Pillow Image to an HTTP PNG Response."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return Response(content=buffer.getvalue(), media_type="image/png")


# -----------------------------------------------------------------------------
# Image Processing Endpoints
# -----------------------------------------------------------------------------
@app.post("/grayscale", summary="Convert image to grayscale")
async def grayscale(image: UploadFile = File(...)) -> Response:
    img = await get_image_from_file(image)
    gray = img.convert("L")
    return image_to_response(gray)


@app.post("/brightness", summary="Adjust image brightness")
async def brightness(request: Request, image: UploadFile = File(...)) -> Response:
    params = await get_params_from_request(request)
    brightness_val = float(params.get("brightness", 1.0))

    img = await get_image_from_file(image)
    enhancer = ImageEnhance.Brightness(img)
    enhanced = enhancer.enhance(brightness_val)

    return image_to_response(enhanced)


@app.post("/contrast", summary="Adjust image contrast")
async def contrast(request: Request, image: UploadFile = File(...)) -> Response:
    params = await get_params_from_request(request)
    contrast_val = float(params.get("contrast", 1.0))

    img = await get_image_from_file(image)
    enhancer = ImageEnhance.Contrast(img)
    enhanced = enhancer.enhance(contrast_val)

    return image_to_response(enhanced)


@app.post("/rotate", summary="Rotate image by angle (degrees)")
async def rotate(request: Request, image: UploadFile = File(...)) -> Response:
    params = await get_params_from_request(request)
    angle = float(params.get("rotation", 0.0))

    img = await get_image_from_file(image)
    rotated = img.rotate(angle, expand=True)

    return image_to_response(rotated)


# -----------------------------------------------------------------------------
# Mount Gradio App
# -----------------------------------------------------------------------------
demo: gr.Blocks = processor.create_app()
gr.mount_gradio_app(app, demo, path="/image-editor")
