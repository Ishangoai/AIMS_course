import tempfile

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

# import requests

API_URL = "http://0.0.0.0:8080"


def edit_image(image, grayscale, brightness, contrast, rotation, text):
    if image is None:
        return None

    # Convert  numpy array to PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    img = image.convert("RGB")

    if grayscale:
        img = img.convert("L").convert("RGB")

    enhancer_b = ImageEnhance.Brightness(img)
    img = enhancer_b.enhance(brightness)

    enhancer_b = ImageEnhance.Contrast(img)
    img = enhancer_b.enhance(contrast)

    img = img.rotate(rotation, expand=True)

    if text:
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except IOError:
            font = ImageFont.load_default()
        draw.text((100, 100), text, fill=(0, 0, 0), font=font)

    return img


def save_temp_image(img):
    if img is None:
        return None

    # Convert numpy array to PIL if needed
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(temp_file.name)
    return temp_file.name


# Gradio App Definition


with gr.Blocks() as app:
    gr.Markdown("# MLOps Assignment 1")
    gr.Markdown("#### Image Manipulation App")

    with gr.Row(scale=1):
        with gr.Column(scale=1):
            input_image = gr.Image(label="Upload Image", type="pil", width=912, height=512)

            grayscale = gr.Checkbox(label="Convert to Grayscale", value=False, info="Grayscale?")

            # sliders
            brightness_input = gr.Slider(0.5, 1.5, value=1, label="Brightness", info="Choose between 0.5 and 1.5")
            contrast_input = gr.Slider(0.5, 1.5, value=1, label="Contrast", info="Choose between 0.5 and 1.5")
            image_rotation_input = gr.Slider(-180, 180, value=0, label="Rotation", info="Choose between -180 and 180")
            text_input = gr.Textbox(label="Add Text to Image", placeholder="Enter text here...", lines=1)
        # col2 = gr.Markdown("Column 2") #
        with gr.Column(scale=1):
            output_image = gr.Image(label="Edited Image", height=712)
            reset_btn = gr.Button("Reset to Original")
            # download_btn = gr.DownloadButton(label="Download Edited Image")

    inputs = [input_image, grayscale, brightness_input, contrast_input, image_rotation_input, text_input]

    # Automatically update on input changes
    for inp in inputs:
        inp.change(fn=edit_image, inputs=inputs, outputs=output_image)

    # Reset button restores original image and reset settings
    def reset_to_original(image):
        return image, False, 1.0, 1.0, 0, "", image

    reset_btn.click(
        fn=reset_to_original,
        inputs=[input_image],
        outputs=inputs + [output_image]
    )
