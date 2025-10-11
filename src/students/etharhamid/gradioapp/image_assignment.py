# Import Standard library
import tempfile

import gradio as gr
import numpy as np

# Import Third-party
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


# The core function to edit the image
def edit_image(image, grayscale, brightness, contrast, rotation, flip_h, flip_v, blur):

    if image is None:
        return None

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    try:

        edited_image = image.convert("RGB")

        # Apply grayscale if checked
        if grayscale:
            edited_image = edited_image.convert("L").convert("RGB")

        # Adjust brightness
        enhancer_brightness = ImageEnhance.Brightness(edited_image)
        edited_image = enhancer_brightness.enhance(brightness)

        # Adjust contrast
        enhancer_contrast = ImageEnhance.Contrast(edited_image)
        edited_image = enhancer_contrast.enhance(contrast)

        # Rotate the image
        edited_image = edited_image.rotate(rotation, expand=True, fillcolor='black')

        # Flip the image
        if flip_h:
            edited_image = ImageOps.mirror(edited_image)

        if flip_v:
            edited_image = ImageOps.flip(edited_image)

        # Blur the image
        if blur > 0:
            edited_image = edited_image.filter(ImageFilter.GaussianBlur(radius=blur))

        return edited_image

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# Function to reset all controls to their default values
def reset_all(original_image):
    return original_image, False, 1.0, 1.0, 0, False, False, 0, original_image


def save_temp_image(image):
    if image is None:
        return None

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(temp_file.name)
    return temp_file.name


theme = gr.Theme(
    primary_hue="blue",
    secondary_hue="purple",
    neutral_hue="gray"
)

# Building the Gradio Interface
with gr.Blocks(theme=theme) as image_app:
    gr.Markdown('<h1 style="text-align: center;"> Image Editor</h1>')
    gr.Markdown("Upload an image and use the controls to edit it. The edited image will be displayed on the right.")

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Input Image", sources=["upload", "clipboard"])
            grayscale_check = gr.Checkbox(label="Convert to Grayscale", value=False)
            flip_h_check = gr.Checkbox(label="Flip Horizontal", value=False)
            flip_v_check = gr.Checkbox(label="Flip Vertically", value=False)
            brightness_slider = gr.Slider(minimum=0.5, maximum=1.5, value=1.0, label="Brightness")
            contrast_slider = gr.Slider(minimum=0.5, maximum=1.5, value=1.0, label="Contrast")
            rotation_slider = gr.Slider(minimum=-180, maximum=180, value=0, label="Rotation")
            blur_slider = gr.Slider(0, 10, value=0, step=1, label="Blur")

            with gr.Row():
                reset_btn = gr.Button("Reset")
                # The download button is part of the gr.File component

        with gr.Column(scale=2):
            output_image = gr.Image(type="pil", label="Output Image", interactive=False)
            download_btn = gr.DownloadButton(label="Download Edited Image")

    # Define the components that will act as inputs to the edit_image function
    inputs = [input_image, grayscale_check, brightness_slider, contrast_slider, rotation_slider,
    flip_h_check, flip_v_check, blur_slider]

    # When any input component changes, call the edit_image function
    for component in inputs:
        component.change(fn=edit_image, inputs=inputs, outputs=output_image)

    # Define what happens when the reset button is clicked
    reset_btn.click(
        fn=reset_all,
        inputs=[input_image],
        outputs=inputs + [output_image]
    )

    download_btn.click(fn=save_temp_image, inputs=output_image, outputs=download_btn)
