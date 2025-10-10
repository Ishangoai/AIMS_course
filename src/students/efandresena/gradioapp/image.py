# image.py
import tempfile

import gradio as gr
from PIL import ImageEnhance, ImageOps


def process_image(image, brightness=1.0, grayscale=False):
    if image is None:
        return None
    img = image.convert("RGB")
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
    if grayscale:
        img = ImageOps.grayscale(img)
    return img


def build_gradio_app():
    with gr.Blocks() as demo:
        gr.Markdown("## Image Adjustment App ")

        with gr.Row():
            with gr.Column():
                img_input = gr.Image(label="Upload an image", type="pil")
                brightness_slider = gr.Slider(0.1, 2, value=1, step=0.05, label="Brightness")
                grayscale_toggle = gr.Checkbox(label="Grayscale")
            with gr.Column():
                img_output = gr.Image(label="Adjusted Image")
                download_btn = gr.DownloadButton(label="Download Adjusted Image")

        def update_and_return(image, brightness, grayscale):
            if image is None:
                return None, None
            img = process_image(image, brightness, grayscale)

            # Save image to a temporary file for download
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            img.save(temp_file.name)
            temp_file.close()

            return img, temp_file.name

        img_input.change(
            update_and_return, 
            [img_input, brightness_slider, grayscale_toggle], 
            [img_output, download_btn]
            )

        brightness_slider.change(
            update_and_return, 
            [img_input, brightness_slider, grayscale_toggle], 
            [img_output, download_btn]
            )

        grayscale_toggle.change(
            update_and_return, 
            [img_input, brightness_slider, grayscale_toggle], 
            [img_output, download_btn]
            )

    return demo
