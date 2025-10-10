import io

import gradio as gr
import requests as req
from PIL import Image

API_BASE_URL = "http://0.0.0.0:8080"


def _send_image_to_api(endpoint: str, image: Image.Image, params: dict = None) -> Image.Image:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    buffered.seek(0)

    files = {'image': ('image.png', buffered, 'image/png')}
    data = params or {}

    response = req.post(f"{API_BASE_URL}/{endpoint}", files=files, data=data)

    if response.status_code != 200:
        raise Exception(f"API request failed: {response.status_code} - {response.text}")

    return Image.open(io.BytesIO(response.content))


def to_grayscale(image: Image.Image) -> Image.Image:
    return _send_image_to_api("/grayscale", image)


def adjust_brightness(image: Image.Image, brightness: float) -> Image.Image:
    return _send_image_to_api("/brightness", image, {"brightness": str(brightness)})


def adjust_contrast(image: Image.Image, contrast: float) -> Image.Image:
    return _send_image_to_api("/contrast", image, {"contrast": str(contrast)})


def rotate_image(image: Image.Image, rotation: float) -> Image.Image:
    return _send_image_to_api("/rotate", image, {"rotation": str(rotation)})


# Unified apply function: handles both image & parameter histories
def apply_modification(
    img,
    img_hist,
    param_hist,
    endpoint_fn,
    current_params,
    updated_param_name=None,
    updated_value=None,
):
    if img is None:
        return None, img_hist, param_hist

    # Create new param snapshot (copy last or defaults)
    new_params = param_hist[-1].copy() if param_hist else {"brightness": 1.0, "contrast": 1.0, "rotation": 0}

    # Update only the changed parameter
    if updated_param_name:
        new_params[updated_param_name] = updated_value

    # Apply the image transformation
    new_img = endpoint_fn(img, updated_value) if updated_value is not None else endpoint_fn(img)

    # Save both image & parameter history
    img_hist.append(img.copy())
    param_hist.append(new_params.copy())

    return new_img, img_hist, param_hist


def revert_change(img_hist, param_hist):
    if not img_hist or len(img_hist) <= 1:
        return img_hist[0] if img_hist else None, img_hist, param_hist, 1.0, 1.0, 0

    img_hist.pop()
    param_hist.pop()

    prev_img = img_hist[-1]
    prev_params = param_hist[-1]

    return (
        prev_img,
        img_hist,
        param_hist,
        prev_params["brightness"],
        prev_params["contrast"],
        prev_params["rotation"]
    )


def restore_image(img_hist, param_hist):
    if not img_hist:
        return None, [], [], 1.0, 1.0, 0

    orig_img = img_hist[0]
    orig_params = {"brightness": 1.0, "contrast": 1.0, "rotation": 0}

    return orig_img, [orig_img.copy()], [orig_params], 1.0, 1.0, 0


def create_app():
    with gr.Blocks() as demo:
        gr.Markdown("## 🖼️ Image Editor")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Upload Image", type="pil")
                revert_btn = gr.Button("Revert")
                restore_btn = gr.Button("Revert all changes")
                gr.DownloadButton(label="DOWNLOAD IMAGE")
            with gr.Column():
                grayscale = gr.Checkbox(label="Convert to Grayscale")
                brightness = gr.Slider(0.5, 1.5, value=1.0, label="Brightness")
                contrast = gr.Slider(0.5, 1.5, value=1.0, label="Contrast")
                rotation = gr.Slider(-180, 180, value=0, label="Rotation (Degrees)")

        # Two states: one for image history, one for parameter history
        image_history = gr.State([])
        param_history = gr.State([])

        # Grayscale modification
        grayscale.change(
            fn=lambda img, imgs, params: apply_modification(
                img, imgs, params, to_grayscale, None
            ),
            inputs=[input_image, image_history, param_history],
            outputs=[input_image, image_history, param_history]
        )

        # Brightness modification
        brightness.change(
            fn=lambda img, val, imgs, params: apply_modification(
                img, imgs, params, adjust_brightness, params, "brightness", val
            ),
            inputs=[input_image, brightness, image_history, param_history],
            outputs=[input_image, image_history, param_history]
        )

        # Contrast modification
        contrast.change(
            fn=lambda img, val, imgs, params: apply_modification(
                img, imgs, params, adjust_contrast, params, "contrast", val
            ),
            inputs=[input_image, contrast, image_history, param_history],
            outputs=[input_image, image_history, param_history]
        )

        # Rotation modification
        rotation.change(
            fn=lambda img, val, imgs, params: apply_modification(
                img, imgs, params, rotate_image, params, "rotation", val
            ),
            inputs=[input_image, rotation, image_history, param_history],
            outputs=[input_image, image_history, param_history]
        )

        # Revert (to previous image & params)
        revert_btn.click(
            fn=revert_change,
            inputs=[image_history, param_history],
            outputs=[input_image, image_history, param_history, brightness, contrast, rotation]
        )

        # Restore (to original)
        restore_btn.click(
            fn=restore_image,
            inputs=[image_history, param_history],
            outputs=[input_image, image_history, param_history, brightness, contrast, rotation]
        )

    return demo
