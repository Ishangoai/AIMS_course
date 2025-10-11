import os
import tempfile

import gradio as gr
from PIL import Image, ImageEnhance


def adjust_brightness(img: Image.Image, brightness: float) -> Image.Image:
    """
    Adjust brightness
    Args:
        img (Image.Image): The image to adjust
        brightness (float): brightness factor

    Returns:
        Image.Image: brightness adjusted image
    """
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(brightness)


def adjust_contrast(img: Image.Image, contrast: float) -> Image.Image:
    """
    Adjust contrast
    Args:
        img (Image.Image):  The image to adjust
        contrast (float): contrast factor
    Returns:
        Image.Image: contrast adjusted image
    """
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(contrast)


def rotate_image(img: Image.Image, angle: float) -> Image.Image:
    """
    Rotate an image by a given angle.

    Args:
        img (Image.Image): The image to rotate
        angle (float): The rotation angle in degrees, counter-clockwise

    Returns:
        Image.Image: A new rotated image.
    """
    return img.rotate(angle, expand=True)


def to_grayscale(img: Image.Image) -> Image.Image:
    """
    Transform image to grayscale
    Args:
        img (Image.Image):  The image to adjust
    Returns:
        Image.Image: grayscale image
    """
    return img.convert("L")


def combined_effects(
    img: Image.Image, brightness: float, contrast: float, angle: float, is_grayscale: bool = False
) -> Image.Image:
    """
    Transform image
    Args:
        img (Image.Image):  The image to adjust
        brightness (float): brightness factor
        contrast (float): contrast factor
        angle (float): The rotation angle in degrees, counter-clockwise
        is_grayscale(boolean)

    Returns:
        Image.Image: contrast adjusted image
    """
    img = adjust_brightness(img, brightness)
    img = adjust_contrast(img, contrast)
    img = rotate_image(img, angle)
    if is_grayscale:
        img = to_grayscale(img)
    return img


def save_image(img: Image.Image, path: str = "") -> str:
    """Download Image

    Args:
        img (Image.Image): Image to be downloaded. Defaults to None.

        path (str): Download path, Defaults to ""
    Returns:
        str: Download path
    """
    if path:
        save_path = os.path.join(path, "modified_image.png")
    else:
        temp_dir = tempfile.mkdtemp()
        save_path = os.path.join(temp_dir, "modified_image.png")
    img.save(save_path)
    return save_path


# --- Gradio interface ---

with gr.Blocks(css="body {background: #f2f7ff;}") as image_transformation:
    gr.Markdown("## Image Operations\nUpload an image, apply effects"
                "(brightness, contrast, rotation, grayscale), or reset to the original image."
                )

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Image")

    with gr.Row():
        grayscale = gr.Checkbox(False, label="Convert to Grayscale")
        brightness = gr.Slider(0.5, 1.5, value=1.0, step=0.1, label="Brightness")
        contrast = gr.Slider(0.5, 1.5, value=1.0, step=0.1, label="Contrast")
        rotate = gr.Slider(-180, 180, value=0, step=1, label="Rotation (degrees)")
        
    output_image = gr.Image(type="pil", label="Final Combined Output", interactive=False)
    Modify = gr.Button("Modify Image")

    Modify.click(
        combined_effects,
        inputs=[image_input, brightness, contrast, rotate, grayscale],
        outputs=output_image
    )

    # --- Bouton Reset ---
    gr.Markdown("### Reset to Original Image")
    reset_button = gr.Button("Reset to Original")
    reset_button.click(
        fn=lambda x: x,
        inputs=image_input,
        outputs=[output_image],
    )

    # Save button
    gr.Markdown("### Download Image")
    save_button = gr.Button("Save Image")
    file_output = gr.File(label="Download your edited image")
    save_button.click(
        save_image,
        inputs=output_image,
        outputs=file_output
    )
