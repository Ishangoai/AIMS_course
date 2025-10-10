import colorsys
import os
import tempfile
from typing import Optional, Tuple

import gradio as gr
from PIL import Image, ImageEnhance


def adjust_hue(image: Image.Image, hue_shift: float) -> Image.Image:
    image = image.convert("RGB")
    pixels = image.load()
    width, height = image.size
    for x in range(width):
        for y in range(height):
            r, g, b = pixels[x, y]
            h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
            h = (h + hue_shift / 360) % 1
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            pixels[x, y] = (int(r * 255), int(g * 255), int(b * 255))
    return image


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


with gr.Blocks(theme=gr.themes.Default()) as image_app:
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
        inputs=state_original,
        outputs=[output_image, grayscale, brightness, contrast, hue, saturation, sharpness, rotate, download_button],
    )


if __name__ == "__main__":
    image_app.launch()
