from __future__ import annotations

import io
import os
import tempfile
from typing import Callable, Dict, List, Optional, Tuple

import gradio as gr
import requests as req
from PIL import Image

API_BASE_URL = "http://0.0.0.0:8080"


def _send_image_to_api(
    endpoint: str,
    image: Image.Image,
    params: Optional[Dict[str, str]] = None,
) -> Image.Image:
    """Send an image to the backend API and return the processed result."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    buffered.seek(0)
    files = {"image": ("image.png", buffered, "image/png")}
    data = params or {}
    response = req.post(f"{API_BASE_URL}/{endpoint}", files=files, data=data)
    if response.status_code != 200:
        raise RuntimeError(
            f"API request failed: {response.status_code} - {response.text}"
        )
    return Image.open(io.BytesIO(response.content))


def to_grayscale(image: Image.Image) -> Image.Image:
    """Convert the image to grayscale via the API."""
    return _send_image_to_api("grayscale", image)


def adjust_brightness(image: Image.Image, brightness: float) -> Image.Image:
    """Adjust brightness of the image via the API."""
    return _send_image_to_api("brightness", image, {"brightness": str(brightness)})


def adjust_contrast(image: Image.Image, contrast: float) -> Image.Image:
    """Adjust contrast of the image via the API."""
    return _send_image_to_api("contrast", image, {"contrast": str(contrast)})


def rotate_image(image: Image.Image, rotation: float) -> Image.Image:
    """Rotate the image via the API."""
    return _send_image_to_api("rotate", image, {"rotation": str(rotation)})


ImageHistory = List[Image.Image]
ParamHistory = List[Dict[str, float]]
ApplyResult = Tuple[Optional[Image.Image], ImageHistory, ParamHistory]


def apply_modification(
    img: Optional[Image.Image],
    img_hist: ImageHistory,
    param_hist: ParamHistory,
    endpoint_fn: Callable[..., Image.Image],
    current_params: Optional[Dict[str, float]] = None,
    updated_param_name: Optional[str] = None,
    updated_value: Optional[float] = None,
) -> ApplyResult:
    """Apply a transformation, update history and parameters."""
    if img is None:
        return None, img_hist, param_hist

    new_params: Dict[str, float] = (
        param_hist[-1].copy()
        if param_hist
        else {"brightness": 1.0, "contrast": 1.0, "rotation": 0.0}
    )

    if updated_param_name is not None and updated_value is not None:
        new_params[updated_param_name] = updated_value

    if updated_value is not None:
        new_img = endpoint_fn(img, updated_value)
    else:
        new_img = endpoint_fn(img)

    img_hist.append(img.copy())
    param_hist.append(new_params.copy())
    return new_img, img_hist, param_hist


def revert_change(
    img_hist: ImageHistory, param_hist: ParamHistory
) -> Tuple[
    Optional[Image.Image],
    ImageHistory,
    ParamHistory,
    float,
    float,
    float,
]:
    """Revert the last applied image modification."""
    if not img_hist or len(img_hist) <= 1:
        if img_hist:
            return img_hist[0], img_hist, param_hist, 1.0, 1.0, 0.0
        return None, img_hist, param_hist, 1.0, 1.0, 0.0

    img_hist.pop()
    param_hist.pop()
    prev_img = img_hist[-1]
    prev_params = param_hist[-1]
    return (
        prev_img,
        img_hist,
        param_hist,
        prev_params.get("brightness", 1.0),
        prev_params.get("contrast", 1.0),
        prev_params.get("rotation", 0.0),
    )


def restore_image(
    img_hist: ImageHistory, param_hist: ParamHistory
) -> Tuple[
    Optional[Image.Image],
    ImageHistory,
    ParamHistory,
    float,
    float,
    float,
]:
    """Restore the image to its original state."""
    if not img_hist:
        return None, [], [], 1.0, 1.0, 0.0

    orig_img = img_hist[0]
    orig_params = {"brightness": 1.0, "contrast": 1.0, "rotation": 0.0}
    return orig_img, [orig_img.copy()], [orig_params], 1.0, 1.0, 0.0


def _prepare_download(image: Optional[Image.Image]) -> str:
    """Prepare the image for download by saving it to a temporary file."""
    if image is None:
        raise gr.Error("No image available to download.")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    image.save(temp_file, format="PNG")
    temp_file.close()

    friendly_name = os.path.join(os.path.dirname(temp_file.name), "edited_image.png")
    os.replace(temp_file.name, friendly_name)

    return friendly_name


def create_app() -> gr.Blocks:
    """Create and return the Gradio image editor interface."""
    with gr.Blocks() as demo:
        gr.Markdown("## 🖼️ Image Editor")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Upload Image", type="pil")
                revert_btn = gr.Button("Revert")
                restore_btn = gr.Button("Revert all changes")
                download_btn = gr.DownloadButton(label="DOWNLOAD IMAGE")

            with gr.Column():
                grayscale = gr.Checkbox(label="Convert to Grayscale")
                brightness = gr.Slider(0.5, 1.5, value=1.0, label="Brightness")
                contrast = gr.Slider(0.5, 1.5, value=1.0, label="Contrast")
                rotation = gr.Slider(-180, 180, value=0, label="Rotation (Degrees)")

        image_history = gr.State([])
        param_history = gr.State([])

        grayscale.change(
            fn=lambda img, imgs, params: apply_modification(
                img, imgs, params, to_grayscale, None
            ),
            inputs=[input_image, image_history, param_history],
            outputs=[input_image, image_history, param_history],
        )

        brightness.change(
            fn=lambda img, val, imgs, params: apply_modification(
                img, imgs, params, adjust_brightness, params, "brightness", val
            ),
            inputs=[input_image, brightness, image_history, param_history],
            outputs=[input_image, image_history, param_history],
        )

        contrast.change(
            fn=lambda img, val, imgs, params: apply_modification(
                img, imgs, params, adjust_contrast, params, "contrast", val
            ),
            inputs=[input_image, contrast, image_history, param_history],
            outputs=[input_image, image_history, param_history],
        )

        rotation.change(
            fn=lambda img, val, imgs, params: apply_modification(
                img, imgs, params, rotate_image, params, "rotation", val
            ),
            inputs=[input_image, rotation, image_history, param_history],
            outputs=[input_image, image_history, param_history],
        )

        revert_btn.click(
            fn=revert_change,
            inputs=[image_history, param_history],
            outputs=[
                input_image,
                image_history,
                param_history,
                brightness,
                contrast,
                rotation,
            ],
        )

        restore_btn.click(
            fn=restore_image,
            inputs=[image_history, param_history],
            outputs=[
                input_image,
                image_history,
                param_history,
                brightness,
                contrast,
                rotation,
            ],
        )

        download_btn.click(
            fn=_prepare_download,
            inputs=[input_image],
            outputs=[download_btn],
        )

    return demo
