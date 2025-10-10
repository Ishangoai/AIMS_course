import gradio as gr
import numpy as np
import scipy.ndimage as si


def img_to_grayscale(image: np.ndarray, to_grayscale: bool):
    if to_grayscale:
        return image.mean(axis=2).astype(int)
    return image


def change_brightness(image: np.ndarray, value: float):
    image = image.astype(np.float32)
    image *= value
    image = np.clip(image, 0, 255)
    image = image.astype(np.uint8)
    return image


def change_contrast(image: np.ndarray, value: float):
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    image = (image - mean) * value + mean
    image = np.clip(image, 0, 255)
    image = image.astype(np.uint8)
    return image


def rotate_image(image: np.ndarray, radius: float):
    image = si.rotate(image, radius, reshape=True)
    return image


def blur_image(image: np.ndarray, sigma: float):
    if len(image.shape) == 2:
        return si.gaussian_filter(image, sigma)
    else:
        return si.gaussian_filter(image, (sigma, sigma, 0))

def transform_image(
    image: np.ndarray | None,
    to_grayscale: bool, 
    brightness: float, 
    contrast: float, 
    rotate: float,
    blur_sigma: float,
):
    if image is None:
        return None
    image = img_to_grayscale(image, to_grayscale)
    image = change_brightness(image, brightness)
    image = change_contrast(image, contrast)
    image = rotate_image(image, rotate)
    image = blur_image(image, blur_sigma)
    return image


with gr.Blocks() as app:
    gr.Markdown("# Image Editor")

    with gr.Row():
        with gr.Column():
            src = gr.Image()
            #src = gr.Image(sources="webcam")

        with gr.Column():
            dst = gr.Image()

    grayscale = gr.Checkbox(False, label="Grayscale")
    brightness = gr.Slider(0.5, 1.5, value=1.0, label="Brightness")
    contrast = gr.Slider(0.5, 1.5, value=1.0, label="Contrast")
    rotate = gr.Slider(-180.0, 180.0, value=0.0, label="Rotate")
    blur = gr.Slider(0, 5, value=0, step=1, label="Blur")

    transforms = [grayscale, brightness, contrast, rotate, blur]

    gr.on(fn=transform_image, inputs=[src, *transforms], outputs=dst)
    #for transform in transforms:
    #    transform.change(transform_image, [src, *transforms], dst)

    #src.stream(transform_image, [src, *transforms], dst, stream_every=0.001)