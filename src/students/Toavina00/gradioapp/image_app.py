import gradio as gr
import numpy as np
from scipy import ndimage


def check_bound(x: int, y: int, x_max: int, y_max: int):
    return (0 <= x < x_max) and (0 <= y < y_max)


def img_to_grayscale(image: np.ndarray, to_grayscale: bool):
    if to_grayscale and (len(image.shape) == 3):
        image = image.dot([0.2989, 0.5870, 0.1140])
    return image


def dithering(image: np.ndarray, dither: bool):
    if not dither:
        return image

    m, n = image.shape[:2]

    # Floyd–Steinberg dithering Algorithm
    for x in range(m):
        for y in range(n):
            oldpixel = image[x, y]
            newpixel = np.where(oldpixel > 0, 1.0, -1.0)
            image[x, y] = newpixel
            quant_error = oldpixel - newpixel

            if check_bound(x, y + 1, m, n):
                image[x, y + 1] += quant_error * 5 / 16
            if check_bound(x + 1, y, m, n):
                image[x + 1, y] += quant_error * 7 / 16
            if check_bound(x + 1, y + 1, m, n):
                image[x + 1, y + 1] += quant_error * 1 / 16
            if check_bound(x - 1, y + 1, m, n):
                image[x - 1, y + 1] += quant_error * 3 / 16

    return image


def invert_color(image: np.ndarray, invert: bool):
    if invert:
        image *= -1
    return image


def flip_vertical(image: np.ndarray, flip: bool):
    if flip:
        n = image.shape[1]
        if len(image.shape) == 2:
            image = image[:, n::-1]
        else:
            image = image[:, n::-1, :]
    return image


def flip_horizontal(image: np.ndarray, flip: bool):
    if flip:
        n = image.shape[0]
        if len(image.shape) == 2:
            image = image[n::-1, :]
        else:
            image = image[n::-1, :, :]
    return image


def change_brightness(image: np.ndarray, value: float):
    image *= value
    image = np.clip(image, -1.0, 1.0)
    return image


def change_contrast(image: np.ndarray, value: float):
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    image = (image - mean) * value + mean
    image = np.clip(image, -1.0, 1.0)
    return image


def rotate_image(image: np.ndarray, radius: float):
    image = ndimage.rotate(image, radius, reshape=True)
    return image


def blur_image(image: np.ndarray, sigma: float):
    if len(image.shape) == 2:
        return ndimage.gaussian_filter(image, sigma)
    else:
        return ndimage.gaussian_filter(image, (sigma, sigma, 0))


def normalize_image(image: np.ndarray):
    image = image.astype(np.float64)
    image -= 255.0 / 2
    image /= 255.0 / 2
    return image


def denormalize_image(image: np.ndarray):
    image *= 255.0 / 2
    image += 255.0 / 2
    image = np.clip(image, 0, 255)
    image = image.astype(int)
    return image


def transform_image(
    image: np.ndarray | None,
    to_grayscale: bool,
    invert: bool,
    flip_h: bool,
    flip_v: bool,
    dither: bool,
    brightness: float,
    contrast: float,
    rotate: float,
    blur_sigma: float,
):
    if image is None:
        return None
    image = normalize_image(image)  # Normalize image to pixels of (-1.0, 1.0) for numerical stability
    image = change_brightness(image, brightness)
    image = change_contrast(image, contrast)
    image = img_to_grayscale(image, to_grayscale)
    image = blur_image(image, blur_sigma)
    image = flip_horizontal(image, flip_h)
    image = flip_vertical(image, flip_v)
    image = invert_color(image, invert)
    image = dithering(image, dither)
    image = rotate_image(image, rotate)
    image = denormalize_image(image)  # Restore image to pixels of (0, 255)
    return image


with gr.Blocks() as app:
    gr.Markdown("# Image Editor")

    with gr.Row():
        with gr.Column():
            src = gr.Image()

        with gr.Column():
            dst = gr.Image()

    with gr.Row():
        grayscale = gr.Checkbox(False, label="Grayscale")
        invert = gr.Checkbox(False, label="Invert Color")
        flip_h = gr.Checkbox(False, label="Flip Horizontal")
        flip_v = gr.Checkbox(False, label="Flip Vertical")
        dither = gr.Checkbox(False, label="Dithering")

    brightness = gr.Slider(0.5, 1.5, value=1.0, label="Brightness")
    contrast = gr.Slider(0.5, 1.5, value=1.0, label="Contrast")
    rotate = gr.Slider(-180.0, 180.0, value=0.0, label="Rotate")
    blur = gr.Slider(0, 5, value=0, step=1, label="Blur")

    with gr.Row():
        reset_btn = gr.Button("Reset")
        download_btn = gr.Button("Download")

    transforms = [
        grayscale,
        invert,
        flip_h,
        flip_v,
        dither,
        brightness,
        contrast,
        rotate,
        blur,
    ]  # Keep a list of transformations to facilitate event handling
    initial_value = [trans.value for trans in transforms]  # Keep track of the initial value for reset event

    gr.on(fn=transform_image, inputs=[src, *transforms], outputs=dst)

    reset_btn.click(lambda: initial_value, outputs=[*transforms])
