import os
import tempfile
from typing import Optional, Tuple

import gradio as gr
import numpy as np
from PIL import Image, ImageEnhance


def adjust_hue(image: Image.Image, hue_shift: float) -> Image.Image:
    image = image.convert("RGB")
    arr = np.asarray(image).astype(np.float32) / 255.0  # shape (H, W, 3), values in [0,1]
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    # Vectorized RGB to HSV
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    deltac = maxc - minc
    s = np.where(maxc == 0, 0, deltac / maxc)
    # Hue calculation
    rc = (maxc - r) / (deltac + 1e-10)
    gc = (maxc - g) / (deltac + 1e-10)
    bc = (maxc - b) / (deltac + 1e-10)
    h = np.zeros_like(maxc)
    h[(maxc == r) & (deltac != 0)] = (bc - gc)[(maxc == r) & (deltac != 0)]
    h[(maxc == g) & (deltac != 0)] = 2.0 + (rc - bc)[(maxc == g) & (deltac != 0)]
    h[(maxc == b) & (deltac != 0)] = 4.0 + (gc - rc)[(maxc == b) & (deltac != 0)]
    h = (h / 6.0) % 1.0
    # Adjust hue
    h = (h + hue_shift / 360.0) % 1.0
    # Vectorized HSV to RGB
    i = np.floor(h * 6.0).astype(int)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    conditions = [
        (i == 0),
        (i == 1),
        (i == 2),
        (i == 3),
        (i == 4),
        (i == 5),
    ]
    rgb = np.zeros(arr.shape)
    rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v])
    rgb[..., 1] = np.select(conditions, [t, v, v, q, p, p])
    rgb[..., 2] = np.select(conditions, [p, p, t, v, v, q])
    rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")


def _apply_edits(
    image: Optional[Image.Image],
    state_original: Optional[Image.Image],
    grayscale: bool,
    brightness: float,
    contrast: float,
    hue: float,
    saturation: float,
    sharpness: float,
    rotate: float,
) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
    """
    Apply the selected edits to *image* and return the updated original
    (for state) and the edited image.
    """
    if image is not None:
        state_original = image

    if state_original is None:
        return None, None

    edited = state_original.copy()

    if hue != 0:
        edited = adjust_hue(edited, hue)

    edited = ImageEnhance.Color(edited).enhance(saturation)
    edited = ImageEnhance.Brightness(edited).enhance(brightness)
    edited = ImageEnhance.Contrast(edited).enhance(contrast)
    edited = ImageEnhance.Sharpness(edited).enhance(sharpness)

    if grayscale:
        edited = edited.convert("L")

    if rotate != 0:
        edited = edited.rotate(rotate, expand=True)

    return state_original, edited


def _reset(
    state_original: Optional[Image.Image],
) -> Tuple[Optional[Image.Image], bool, float, float, float, float, float, float, Optional[str]]:
    """Return the original image for the preview and reset controls."""
    path = _prepare_download(state_original)
    return state_original, False, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, path


def _prepare_download(image: Optional[Image.Image]) -> Optional[str]:
    """Return a path suitable for download."""
    if image is None:
        return None

    fd, path = tempfile.mkstemp(suffix=".png")
    with os.fdopen(fd, "wb") as f:
        image.save(f, format="PNG")
    return path


def upload_fn(path: Optional[str]) -> Tuple[Optional[Image.Image], Optional[Image.Image],
 bool, float, float, float, float, float, float, Optional[str]]:
    if path is None:
        return None, None, False, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, None
    image = Image.open(path)
    dl_path = _prepare_download(image)
    return image, image, False, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, dl_path


with gr.Blocks() as image_app:
    gr.Markdown("# Image Editor")

    state_original = gr.State()

    with gr.Row():
        with gr.Column(scale=1):
            upload_button = gr.UploadButton(
                "Upload Image", file_types=["image"], file_count="single"
            )
            grayscale = gr.Checkbox(label="Grayscale")
            brightness = gr.Slider(
                minimum=0.5, maximum=1.5, value=1.0, label="Brightness"
            )
            contrast = gr.Slider(
                minimum=0.5, maximum=1.5, value=1.0, label="Contrast"
            )
            hue = gr.Slider(
                minimum=-180, maximum=180, value=0, label="Hue"
            )
            saturation = gr.Slider(
                minimum=0, maximum=3, value=1, label="Saturation"
            )
            sharpness = gr.Slider(
                minimum=0, maximum=3, value=1, label="Sharpness"
            )
            rotate = gr.Slider(
                minimum=-180, maximum=180, value=0, label="Rotate"
            )
            with gr.Row():
                reset_button = gr.Button("Reset")
                download_button = gr.DownloadButton("Download")

        with gr.Column(scale=4):
            output_image = gr.Image(
                label="Edited image",
                type="pil",
                show_label=True,
                height=600,
                interactive=False,
            )

    def update(
        orig_state: Optional[Image.Image],
        gray: bool,
        bright: float,
        cont: float,
        hue_val: float,
        sat: float,
        sharp: float,
        rot: float,
    ) -> Tuple[Optional[Image.Image], Optional[Image.Image], Optional[str]]:
        orig_state, edited = _apply_edits(None, orig_state, gray, bright, cont, hue_val, sat, sharp, rot)
        dl_path = _prepare_download(edited)
        return orig_state, edited, dl_path

    # Run the edit routine whenever a control changes
    for component in (grayscale, brightness, contrast, hue, saturation, sharpness, rotate):
        component.change(
            fn=update,
            inputs=[state_original, grayscale, brightness, contrast, hue, saturation, sharpness, rotate],
            outputs=[state_original, output_image, download_button],
        )

    # Upload new image and reset controls
    upload_button.upload(
        fn=upload_fn,
        inputs=[upload_button],
        outputs=[
            state_original,
            output_image,
            grayscale,
            brightness,
            contrast,
            hue,
            saturation,
            sharpness,
            rotate,
            download_button,
        ],
    )

    # Reset: show original image and reset controls
    reset_button.click(
        fn=_reset,
        inputs=[state_original],
        outputs=[output_image, grayscale, brightness, contrast, hue, saturation, sharpness, rotate, download_button],
    )


if __name__ == "__main__":
    image_app.launch()
