import tempfile

import gradio as gr
import numpy as np
from PIL import Image, ImageEnhance


def edit_image(image, grayscale, brightness, contrast, rotation):
    """
    Apply basic edits to an image.

    Parameters
    ----------
    image : PIL.Image.Image | numpy.ndarray | None
        Input image. If a numpy array is provided it will be converted into a PIL Image.
    grayscale : bool
        If True, convert the image to grayscale.
    brightness : float
        Brightness multiplier (1.0 = original).
    contrast : float
        Contrast multiplier (1.0 = original).
    rotation : float
        Degrees to rotate the image (positive = counterclockwise).

    Returns
    -------
    PIL.Image.Image | None
        Edited image or None if input was None.
    """

    if image is None:
        return None

    # Convert numpy array to PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    img = image.convert("RGB")

    if grayscale:
        img = img.convert("L").convert("RGB")

    enhancer_b = ImageEnhance.Brightness(img)
    img = enhancer_b.enhance(brightness)

    enhancer_c = ImageEnhance.Contrast(img)
    img = enhancer_c.enhance(contrast)

    img = img.rotate(rotation, expand=True)

    return img


def save_temp_image(img):
    """
    Save an image to a temporary PNG file and return its file path.

    Parameters
    ----------
    img : PIL.Image.Image | numpy.ndarray | None
        Image to save. If None, returns None.

    Returns
    -------
    str | None
        Full filesystem path to the saved temporary PNG file, or None if img was None.
    """

    if img is None:
        return None

    # Convert numpy array to PIL if needed
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(temp_file.name)
    return temp_file.name


# Gradio app definition
with gr.Blocks(title="Image Editor") as image_app:
    gr.Markdown("## 🖼️ Image Editor\nUpload an image and apply basic edits.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Image", type="pil")
            grayscale = gr.Checkbox(label="Convert to Grayscale", value=False)
            brightness = gr.Slider(0.5, 1.5, 1.0, label="Brightness")
            contrast = gr.Slider(0.5, 1.5, 1.0, label="Contrast")
            rotation = gr.Slider(-180, 180, 0, label="Rotate (degrees)")
            reset_btn = gr.Button("Reset to Original")
        with gr.Column():
            image_output = gr.Image(label="Edited Image")
            download_btn = gr.DownloadButton(label="Download Edited Image")

    inputs = [image_input, grayscale, brightness, contrast, rotation]

    # Automatically update on input changes
    for inp in inputs:
        inp.change(fn=edit_image, inputs=inputs, outputs=image_output)

    # Reset button restores original image and resets settings
    def reset_to_original(image):
        """
        Reset UI widgets to their default values and restore original image.

        Parameters
        ----------
        image : PIL.Image.Image | numpy.ndarray | None
            The original image to restore.

        Returns
        -------
        tuple
            (image, False, 1.0, 1.0, 0, image) corresponding to:
            (image_input, grayscale, brightness, contrast, rotation, image_output)
        """
        return image, False, 1.0, 1.0, 0, image

    reset_btn.click(
        fn=reset_to_original,
        inputs=[image_input],
        outputs=inputs + [image_output],
    )

    # Download button saves edited image to temp file
    download_btn.click(fn=save_temp_image, inputs=image_output, outputs=download_btn)
