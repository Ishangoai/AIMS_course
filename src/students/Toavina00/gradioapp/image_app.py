import gradio as gr
import numpy as np
from scipy.ndimage import rotate as sp_rotate


def img_to_grayscale(image: np.array, to_grayscale: bool):
    if to_grayscale:
        return image.mean(axis=2).astype(int)

    return image


def change_brightness(image: np.array, value: float):
    image = image.astype(np.float32)
    image *= value
    image = np.clip(image, 0, 255)
    image = image.astype(np.uint8)
    return image


def change_contrast(image: np.array, value: float):
    image_float = image.astype(np.float32)
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    image = (image - mean) * value + mean
    image = np.clip(image, 0, 255)
    image = image.astype(np.uint8)
    return image


def rotate_image(image: np.array, radius: float):
    image= sp_rotate(image, radius, reshape=True)
    return image


def transform_image(image: np.array, to_grayscale: bool, brightness: float, contrast: float, rotate: float):
    image = img_to_grayscale(image, to_grayscale)
    image = change_brightness(image, brightness)
    image = change_contrast(image, contrast)
    image = rotate_image(image, rotate)

    return image

def reset_all():
    return None, False, 1.0, 1.0, 0.0


with gr.Blocks() as app:
    gr.Markdown("# Image Editor")

    with gr.Row():
        with gr.Column():
            src = gr.Image()

        with gr.Column():
            dst = gr.Image()


    grayscale = gr.Checkbox(False, label="Grayscale")
    brightness = gr.Slider(0.5, 1.5, value=1.0, label="Brightness")
    contrast = gr.Slider(0.5, 1.5, value=1.0, label="Contrast")
    rotate = gr.Slider(-180.0, 180.0, value=0.0, label="Rotate")

    with gr.Row():
        reset_btn = gr.Button("🔄 Reset")
        download_btn = gr.DownloadButton("⬇️ Download", label="Download Edited Image")


    src.upload(lambda x: x, src, dst)  # Initialize the output to be the uploaded image

    transforms = [grayscale, brightness, contrast, rotate]

    for transform in transforms:
        transform.change(transform_image, [src, *transforms], dst)

    # Reset button 
    reset_btn.click(reset_all, outputs=[src, grayscale, brightness, contrast, rotate])

    # Download button 
    download_btn.click(fn=lambda img: img, inputs=dst, outputs=gr.File())

