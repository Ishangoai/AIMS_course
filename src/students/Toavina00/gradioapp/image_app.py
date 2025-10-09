import gradio as gr
import numpy as np


def img_to_grayscale(image: np.array, to_grayscale: bool):
    if to_grayscale:
        return image.mean(axis=2).astype(int)

    return image


def change_brightness(image: np.array, value: float): 
    image = image.astype(np.float32)
    image *= value
    image = np.clip(image,0,255)
    image = image.astype(np.uint8)
    return image


def change_contrast(image: np.array, value: float):
    image_float = image.astype(np.float32)
    mean = np.mean(image, axis = (0,1), keepdims = True)
    image = (image - mean) * value + mean
    image = np.clip(image,0,255)
    image = image.astype(np.uint8)
    return image


def rotate_image(image: np.array, radius: float):
    k = radius  // 90
    image = np.rot90(img, k=k)
    return image


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

    src.upload(lambda x: x, src, dst) # Initialize the output to be the uploaded image

    grayscale.change(img_to_grayscale, [src, grayscale], dst)
    brightness.change(change_brightness, [src, brightness], dst)
    contrast.change(change_contrast, [src, contrast], dst)
    rotate.change(rotate_image, [src, rotate], dst)
