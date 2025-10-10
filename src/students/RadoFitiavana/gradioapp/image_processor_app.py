import gradio as gr
import requests as req
from PIL import Image
import io

API_BASE_URL = "http://0.0.0.0:8080"

def _send_image_to_api(endpoint: str, image: Image.Image, params: dict = None) -> Image.Image:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    buffered.seek(0)

    files = {'image': ('image.png', buffered, 'image/png')}
    data = params or {}

    response = req.post(f"{API_BASE_URL}{endpoint}", files=files, data=data)

    if response.status_code != 200:
        raise Exception(f"API request failed: {response.status_code} - {response.text}")

    # Convert response content back to a PIL image
    return Image.open(io.BytesIO(response.content))


def to_grayscale(image: Image.Image) -> Image.Image:
    return _send_image_to_api("/grayscale", image)


def adjust_brightness(image: Image.Image, brightness: float) -> Image.Image:
    return _send_image_to_api("/brightness", image, {"brightness": str(brightness)})


def adjust_contrast(image: Image.Image, contrast: float) -> Image.Image:
    return _send_image_to_api("/contrast", image, {"contrast": str(contrast)})


def rotate_image(image: Image.Image, rotation: float) -> Image.Image:
    return _send_image_to_api("/rotate", image, {"rotation": str(rotation)})

def process_all(image, grayscale, brightness, contrast, rotation):
    if image is None:
        return None

    img = image.convert("RGB")  # Ensure starting from RGB

    if grayscale:
        img = to_grayscale(img)

    img = adjust_brightness(img, brightness)
    img = adjust_contrast(img, contrast)
    img = rotate_image(img, rotation)

    return img

def create_app():
    with gr.Blocks() as demo:
        gr.Markdown("## 🖼️ Image Editor")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Upload Image", type="pil")
                grayscale = gr.Checkbox(label="Convert to Grayscale")
                brightness = gr.Slider(0.5, 1.5, value=1.0, label="Brightness")
                contrast = gr.Slider(0.5, 1.5, value=1.0, label="Contrast")
                rotation = gr.Slider(-180, 180, value=0, label="Rotation (Degrees)")
            with gr.Column():
                output_image = gr.Image(label="Output Image")

        inputs = [input_image, grayscale, brightness, contrast, rotation]
        for input_component in inputs:
            input_component.change(fn=process_all, inputs=inputs, outputs=output_image)

    return demo

