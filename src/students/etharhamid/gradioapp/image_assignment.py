# Import Standard library
import tempfile

import gradio as gr
import numpy as np

# Import Third-party
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


# The core function to edit the image
def edit_image(image, grayscale, brightness, contrast, rotation, flip_h, flip_v, blur):
    """
    Applies various editing effects to an input image.

    Args:
        image (PIL.Image.Image or np.ndarray): The input image to be edited.
        grayscale (bool): If True, converts the image to grayscale.
        brightness (float): The brightness enhancement factor. 1.0 means no change.
        contrast (float): The contrast enhancement factor. 1.0 means no change.
        rotation (int): The angle in degrees to rotate the image.
        flip_h (bool): If True, flips the image horizontally.
        flip_v (bool): If True, flips the image vertically.
        blur (int): The radius for the Gaussian blur filter. 0 means no blur.

    Returns:
        PIL.Image.Image or None: The edited image, or None if an error occurs or the input is None.
    """

    # Return None if no image is provided
    if image is None:
        return None

    # Convert NumPy array to PIL Image if necessary
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    try:
        # Ensure the image is in RGB mode for consistent processing
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

        # Flip the image horizontally if checked
        if flip_h:
            edited_image = ImageOps.mirror(edited_image)

        # Flip the image vertically if checked
        if flip_v:
            edited_image = ImageOps.flip(edited_image)

        # Apply Gaussian Blur to the image
        if blur > 0:
            edited_image = edited_image.filter(ImageFilter.GaussianBlur(radius=blur))

        return edited_image

    except Exception as e:
        # Print any errors that occur during image processing
        print(f"An error occurred: {e}")
        return None


# Function to reset all controls to their default values
def reset_all(original_image):
    """
    Resets all the UI controls to their initial default values.

    Args:
        original_image (PIL.Image.Image): The original uploaded image to restore.

    Returns:
        tuple: A tuple containing the default values for all the input and output components.
    """
    return original_image, False, 1.0, 1.0, 0, False, False, 0, original_image


def save_temp_image(image):
    """
    Saves a PIL Image to a temporary file and returns the file path.

    Args:
        image (PIL.Image.Image or np.ndarray): The image to be saved.

    Returns:
        str or None: The file path of the saved temporary image, or None if the input is None.
    """

    if image is None:
        return None

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # delete=False ensures the file is not deleted upon closing
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
    gr.Markdown('<h4 style="text-align: center;"> Upload an image and use the controls to edit it.\
        The edited image will be displayed on the right.</h4>')

    # Create a main row to hold the input controls and the output image
    with gr.Row():
        # Create a column for the input controls
        with gr.Column():

            input_image = gr.Image(type="pil", label="Input Image", sources=["upload", "clipboard"])
            with gr.Group():
                grayscale_check = gr.Checkbox(label="Convert to Grayscale", value=False)
                flip_h_check = gr.Checkbox(label="Flip Horizontal", value=False)
                flip_v_check = gr.Checkbox(label="Flip Vertically", value=False)
                brightness_slider = gr.Slider(minimum=0.5, maximum=1.5, value=1.0, label="Brightness")
                contrast_slider = gr.Slider(minimum=0.5, maximum=1.5, value=1.0, label="Contrast")
                rotation_slider = gr.Slider(minimum=-180, maximum=180, value=0, label="Rotation")
                blur_slider = gr.Slider(0, 10, value=0, step=1, label="Blur")
                reset_btn = gr.Button("Reset")

        # Create a second column for the output image and download button
        with gr.Column():
            output_image = gr.Image(type="pil", label="Output Image", interactive=False, height=700)
            download_btn = gr.DownloadButton(label="Download Edited Image", scale=1)

    # Define the components that will act as inputs to the edit_image function
    inputs = [input_image, grayscale_check, brightness_slider, contrast_slider, rotation_slider,
    flip_h_check, flip_v_check, blur_slider]

    # When any input component changes, call the edit_image function
    for component in inputs:
        component.change(fn=edit_image, inputs=inputs, outputs=output_image)

    # Define the click event for the reset button
    reset_btn.click(
        fn=reset_all,
        inputs=[input_image],
        outputs=inputs + [output_image]
    )

    # Define the click event for the download button
    download_btn.click(fn=save_temp_image, inputs=output_image, outputs=download_btn)
