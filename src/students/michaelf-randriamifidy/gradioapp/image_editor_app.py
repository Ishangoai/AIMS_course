import tempfile
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
from PIL import Image, ImageEnhance, ImageFilter

filters_list = ['BLUR', 'CONTOUR', 'DETAIL', 'SHARPEN', 'EDGE_ENHANCE', 'EDGE_ENHANCE_MORE',
                'EMBOSS', 'FIND_EDGES', 'SMOOTH', 'SMOOTH_MORE', 'GAUSSIAN_BLUR', 'UNSHARP_MASK']


def filter_function(img: Image.Image, filter_type: str) -> Image.Image:
    """
    Apply a specified filter to an image.

    Args:
        img (Image.Image): The  image to be filtered.
        filter_type (FilterType): The type of filter to apply.

    Returns:
        Image.Image: The filtered image. If the filter type is invalid,
        returns the original image.
    """
    filters = {
        "BLUR": ImageFilter.BLUR,
        "CONTOUR": ImageFilter.CONTOUR,
        "DETAIL": ImageFilter.DETAIL,
        "SHARPEN": ImageFilter.SHARPEN,
        "EDGE_ENHANCE": ImageFilter.EDGE_ENHANCE,
        "EDGE_ENHANCE_MORE": ImageFilter.EDGE_ENHANCE_MORE,
        "EMBOSS": ImageFilter.EMBOSS,
        "FIND_EDGES": ImageFilter.FIND_EDGES,
        "SMOOTH": ImageFilter.SMOOTH,
        "SMOOTH_MORE": ImageFilter.SMOOTH_MORE,
        "GAUSSIAN_BLUR": ImageFilter.GaussianBlur(2),  # default radius
        "UNSHARP_MASK": ImageFilter.UnsharpMask(
            radius=2, percent=150, threshold=3
        ),
    }

    chosen_filter = filters.get(filter_type)
    if chosen_filter:
        return img.filter(chosen_filter)
    else:
        return img


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


def upload_image(file_path: str) -> Image.Image:
    """
    Open an image from the given file path.

    Args:
        file_path (str): Path to the image file.

    Returns:
        PIL.Image.Image: The opened image.
    """
    img = Image.open(file_path)
    return img


def wrapped_upload_image(file_path: str) -> Tuple[Image.Image, ...]:
    """
    Open an image from the given file path and return
    a tuple containing the image repeated 4 times.

    Args:
        file_path (str): Path to the image file.

    Returns:
        tuple: A tuple of 4 identical PIL.Image.Image objects.
    """
    img = upload_image(file_path)
    return tuple([img] * 4)


def generate_filename(upload_file: str, suffix: Optional[str] = None) -> str:
    """
    Generate a filename based on the uploaded file + suffix.

    Args:
        upload_file (str): Path to the uploaded file.
        suffix (Optional[str]): suffix to append.

    Returns:
        str: Filename with optional suffix.
    """
    basename = upload_file
    suffix = f"_{suffix}" if suffix else ""
    return f"{basename}_{suffix}"


def save_image(img: Image.Image, filename: str = "", ext: str = ".png") -> str:
    """
    Download Image

    Args:
        img (Image.Image): Image to be downloaded. Defaults to None.

        path (str): filename, Defaults to ".png"
    Returns:
        str: save path
    """
    temp_dir = tempfile.mkdtemp()
    if not filename:
        filename = "modified_image"
    save_path = str(Path(temp_dir, f"{filename}{ext}"))
    img.save(save_path)
    return save_path


# --- Gradio interface ---
with gr.Blocks(css="body {background: #f2f7ff;}") as image_transformation:
    gr.Markdown("# Image Operations\n Upload an image, apply effects\n by Ange and Michael Fitiavana")

    with gr.Row():
        upload_button = gr.File(label="Upload Image", file_types=[".png", ".jpg", ".jpeg"])

    with gr.Tabs():
        with gr.Tab("Adjust Image"):
            gr.Markdown("# Convert image to grayscale, set brightness, contrast, rotation")
            grayscale = gr.Checkbox(False, label="Convert to Grayscale")
            brightness = gr.Slider(0.5, 1.5, value=1.0, step=0.1, label="Brightness")
            contrast = gr.Slider(0.5, 1.5, value=1.0, step=0.1, label="Contrast")
            rotate = gr.Slider(-180, 180, value=0, step=1, label="Rotation(degrees)")
            with gr.Row():
                with gr.Column(scale=2):
                    input_image = gr.Image(type="pil", label="Original Image", interactive=False)
                with gr.Column(scale=2):
                    output_image = gr.Image(type="pil", label="Edited Image", interactive=False)
            Modify = gr.Button("Modify Image")

            Modify.click(
            combined_effects,
            inputs=[input_image, brightness, contrast, rotate, grayscale],
            outputs=output_image
            )

            # --- Reset button ---
            gr.Markdown("Reset to Original Image")
            reset_button = gr.Button("Reset to Original")
            reset_button.click(
            fn=lambda x: x,
            inputs=input_image,
            outputs=[output_image],
            )

            # --- Save button ---
            gr.Markdown("### Download Image")
            save_button = gr.Button("Save Image")
            file_output = gr.File(label="Download your edited image")
            save_button.click(
                save_image,
                inputs=[output_image],
                outputs=file_output
            )

        with gr.Tab(label="Filter Image"):
            gr.Markdown("# Choose filter Option")

            with gr.Row():
                with gr.Column(scale=2):
                    filter_input_image = gr.Image(type="pil", label="Original Image", interactive=False)
                with gr.Column(scale=2):
                    filter_output_image = gr.Image(type='pil', label="Filtered image", interactive=False)
            filter_radio = gr.Radio(filters_list, label="Filters")
            filter_button = gr.Button("Filter")

            filter_button.click(
            filter_function,
            inputs=[input_image, filter_radio],
            outputs=filter_output_image
            )

            # --- Reset button ---
            gr.Markdown("### Reset to Original Image")
            reset_button_filter = gr.Button("Reset to Original")
            reset_button_filter.click(
            fn=lambda x: x,
            inputs=input_image,
            outputs=[filter_output_image],
            )

            # --- Save button ---
            gr.Markdown("### Download Image")
            save_button_filter = gr.Button("Save Image")
            file_output_filter = gr.File(label="Download your filter image")

            save_button_filter.click(
                save_image,
                inputs=[filter_output_image, filter_radio],
                outputs=file_output_filter
            )

    upload_button.upload(fn=wrapped_upload_image, inputs=upload_button,
                    outputs=[input_image, output_image, filter_input_image, filter_output_image])


if __name__ == "__main__":
    image_transformation.launch()
