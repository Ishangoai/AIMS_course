"""
Image Editor Application.

A Gradio-based web application for basic image editing and text overlay.
Supports brightness/contrast adjustment, rotation, grayscale conversion,
and customizable text overlays.
"""

import os
import re
import tempfile
from typing import Optional, Tuple, Union

import gradio as gr
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

# Constants
DEFAULT_TEXT_POSITIONS: dict[str, float] = {
    "Top": 0.05,
    "Center": 0.5,
    "Bottom": 0.9
}
DEFAULT_COLOR: Tuple[int, int, int] = (255, 255, 255)
FONT_PATH: str = os.path.join(
    os.path.dirname(ImageFont.__file__), "Fonts/DejaVuSans.ttf"
)


def parse_color(color: Optional[str]) -> Tuple[int, int, int]:
    """
    Parse color string from various formats to RGB tuple.

    Supports hex (#RGB, #RRGGBB) and functional (rgb(), rgba()) formats.
    Returns white (255, 255, 255) as fallback for invalid inputs.

    Args:
        color: Color string in hex, rgb(), or rgba() format.
               Can be None or empty string.

    Returns:
        RGB color tuple with values in range [0, 255].

    Examples:
        >>> parse_color("#FF0000")
        (255, 0, 0)
        >>> parse_color("rgba(128.5, 64.2, 32.8, 1)")
        (128, 64, 33)
    """
    if not color:
        return DEFAULT_COLOR

    color = str(color).strip()

    # Hex format: #RGB or #RRGGBB
    if color.startswith("#"):
        color = color.lstrip("#")
        if len(color) == 3:
            color = "".join([c * 2 for c in color])
        if len(color) == 6:
            try:
                return (
                    int(color[0:2], 16),
                    int(color[2:4], 16),
                    int(color[4:6], 16)
                )
            except ValueError:
                pass

    # RGB/RGBA format: rgb(R,G,B) or rgba(R,G,B,A)
    match = re.match(
        r"rgba?\(([\d.]+),\s*([\d.]+),\s*([\d.]+)(?:,\s*[\d.]+)?\)",
        color
    )
    if match:
        return (
            int(round(float(match.group(1)))),
            int(round(float(match.group(2)))),
            int(round(float(match.group(3))))
        )

    return DEFAULT_COLOR


def load_font(font_size: int) -> Union[ImageFont.FreeTypeFont, ImageFont.ImageFont]:
    """
    Load TrueType font with specified size.

    Attempts to load DejaVuSans font from Pillow's font directory.
    Falls back to default bitmap font if TrueType font is unavailable.

    Args:
        font_size: Font size in points.

    Returns:
        Font object ready for text rendering.
    """
    try:
        return ImageFont.truetype(FONT_PATH, font_size)
    except (OSError, IOError):
        return ImageFont.load_default()


def to_pil_image(image: Union[NDArray[np.uint8], Image.Image]) -> Image.Image:
    """
    Convert input image to PIL RGB format.

    Handles both numpy arrays (from Gradio) and PIL Images.
    Ensures output is always in RGB mode for consistent processing.

    Args:
        image: Input image as numpy array or PIL Image.

    Returns:
        PIL Image in RGB mode.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    return image.convert("RGB")


def apply_transformations(
    img: Image.Image,
    grayscale: bool,
    brightness: float,
    contrast: float,
    rotation: float
) -> Image.Image:
    """
    Apply color and geometric transformations to image.

    Transformations are applied in order: grayscale, brightness,
    contrast, rotation. Rotation expands canvas to fit rotated image.

    Args:
        img: Input PIL Image in RGB mode.
        grayscale: If True, convert image to grayscale.
        brightness: Brightness multiplier (1.0 = original).
        contrast: Contrast multiplier (1.0 = original).
        rotation: Rotation angle in degrees (positive = counter-clockwise).

    Returns:
        Transformed PIL Image.
    """
    if grayscale:
        img = img.convert("L").convert("RGB")

    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)

    if rotation != 0:
        img = img.rotate(rotation, expand=True)

    return img


def calculate_text_position(
    img_size: Tuple[int, int],
    text_size: Tuple[int, int],
    use_custom: bool,
    x: float,
    y: float,
    position: str
) -> Tuple[int, int]:
    """
    Calculate text overlay position based on user preferences.

    Supports both preset positions (Top, Center, Bottom) and custom
    coordinates. Ensures text stays within image bounds.

    Args:
        img_size: Image dimensions as (width, height).
        text_size: Text dimensions as (width, height).
        use_custom: If True, use custom x/y coordinates.
        x: Custom x-coordinate (ignored if use_custom is False).
        y: Custom y-coordinate (ignored if use_custom is False).
        position: Preset position key ("Top", "Center", or "Bottom").

    Returns:
        Text position as (x, y) coordinates for top-left corner.
        Values are clamped to keep text within image bounds.
    """
    img_width, img_height = img_size
    text_width, text_height = text_size

    if use_custom:
        x_pos = min(max(0, int(x)), img_width - text_width)
        y_pos = min(max(0, int(y)), img_height - text_height)
    else:
        x_pos = (img_width - text_width) // 2
        y_rel = DEFAULT_TEXT_POSITIONS.get(position, 0.05)
        y_pos = max(0, min(int(img_height * y_rel), img_height - text_height))

    return x_pos, y_pos


def add_text(
    img: Image.Image,
    text: str,
    color: str,
    font_size: int,
    use_custom: bool,
    x: float,
    y: float,
    position: str
) -> Image.Image:
    """
    Add text overlay to image at specified position.

    Renders text with specified font size and color. Position is calculated
    based on either custom coordinates or preset positions.

    Args:
        img: PIL Image to add text to.
        text: Text string to render.
        color: Text color in any format supported by parse_color().
        font_size: Font size in points.
        use_custom: If True, use custom positioning.
        x: Custom x-coordinate.
        y: Custom y-coordinate.
        position: Preset position ("Top", "Center", "Bottom").

    Returns:
        PIL Image with text overlay applied.
    """
    draw = ImageDraw.Draw(img)
    font = load_font(font_size)

    # Get text dimensions
    bbox = draw.textbbox((0, 0), text, font=font)
    text_size = (int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1]))

    # Calculate position
    x_pos, y_pos = calculate_text_position(
        (img.width, img.height), text_size, use_custom, x, y, position
    )

    # Draw text
    draw.text((x_pos, y_pos), text, font=font, fill=parse_color(color))
    return img


def edit_image(
    image: Optional[Union[NDArray[np.uint8], Image.Image]],
    grayscale: bool,
    brightness: float,
    contrast: float,
    rotation: float,
    text: str = "",
    use_custom_pos: bool = False,
    x: float = 0,
    y: float = 0,
    position: str = "Top",
    color: str = "#FFFFFF",
    font_size: int = 30
) -> Optional[Image.Image]:
    """
    Apply all image edits and text overlay.

    Main processing function that coordinates all image transformations
    and text overlay operations.

    Args:
        image: Input image (numpy array or PIL Image). Can be None.
        grayscale: Convert to grayscale.
        brightness: Brightness adjustment factor.
        contrast: Contrast adjustment factor.
        rotation: Rotation angle in degrees.
        text: Text to overlay (empty string = no text).
        use_custom_pos: Use custom text positioning.
        x: Custom x-coordinate for text.
        y: Custom y-coordinate for text.
        position: Preset text position.
        color: Text color.
        font_size: Text font size in points.

    Returns:
        Edited PIL Image, or None if input image is None.
    """
    if image is None:
        return None

    img = to_pil_image(image)
    img = apply_transformations(img, grayscale, brightness, contrast, rotation)

    if text:
        img = add_text(img, text, color, font_size, use_custom_pos, x, y, position)

    return img


def save_temp_image(
    img: Optional[Union[NDArray[np.uint8], Image.Image]]
) -> Optional[str]:
    """
    Save image to temporary file for download.

    Creates a temporary PNG file that can be downloaded by the user.
    The file is not automatically deleted.

    Args:
        img: Image to save (PIL Image or numpy array). Can be None.

    Returns:
        Path to temporary file, or None if input is None.
    """
    if img is None:
        return None
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(temp_file.name)
    return temp_file.name


def reset_to_original(
    image: Optional[Union[NDArray[np.uint8], Image.Image]]
) -> Tuple[
    Optional[Union[NDArray[np.uint8], Image.Image]],
    bool, float, float, float,
    str, bool, float, float,
    str, str, int,
    Optional[Union[NDArray[np.uint8], Image.Image]]
]:
    """
    Reset all controls to default values and restore original image.

    Returns default values for all UI controls in the order they appear
    in the inputs list, plus the original image for the output display.

    Args:
        image: Original input image.

    Returns:
        Tuple of default values for all controls plus original image.
    """
    return image, False, 1.0, 1.0, 0, "", False, 50, 50, "Top", "#FFFFFF", 30, image


# Gradio Interface
with gr.Blocks(title="Image Editor") as image_app:
    gr.Markdown("## 🖼️ Online Image Editor: Designed by Ghaida and Hassan")

    with gr.Row():
        # Controls Column
        with gr.Column(scale=1):
            image_input = gr.Image(label="Upload Image", type="pil", height=300)

            with gr.Accordion("Basic Edits", open=True):
                grayscale = gr.Checkbox(label="Convert to Grayscale", value=False)
                brightness = gr.Slider(0.5, 1.5, 1.0, label="Brightness")
                contrast = gr.Slider(0.5, 1.5, 1.0, label="Contrast")
                rotation = gr.Slider(-180, 180, 0, label="Rotate (degrees)")

            with gr.Accordion("Text Overlay", open=True):
                text_input = gr.Textbox(label="Text", placeholder="Enter text here")
                with gr.Row():
                    color_picker = gr.ColorPicker(value="#FFFFFF", label="Color")
                    font_size_slider = gr.Slider(10, 100, 30, step=1, label="Size")
                position_dropdown = gr.Dropdown(
                    list(DEFAULT_TEXT_POSITIONS.keys()), value="Top", label="Position"
                )
                use_custom_pos = gr.Checkbox(label="Custom Position", value=False)
                with gr.Row(visible=False) as custom_pos_row:
                    x_slider = gr.Slider(0, 512, 50, label="X")
                    y_slider = gr.Slider(0, 512, 50, label="Y")

            with gr.Row():
                reset_btn = gr.Button("Reset", variant="secondary")
                download_btn = gr.DownloadButton("Download", variant="primary")

        # Preview Column
        with gr.Column(scale=1):
            image_output = gr.Image(label="Preview", height=600, interactive=False)

    # Define inputs list
    inputs = [
        image_input, grayscale, brightness, contrast, rotation,
        text_input, use_custom_pos, x_slider, y_slider,
        position_dropdown, color_picker, font_size_slider
    ]

    # Event Handlers
    use_custom_pos.change(
        fn=lambda x: gr.update(visible=x),
        inputs=use_custom_pos,
        outputs=custom_pos_row
    )

    for inp in inputs:
        inp.change(fn=edit_image, inputs=inputs, outputs=image_output)

    reset_btn.click(
        fn=reset_to_original,
        inputs=[image_input],
        outputs=inputs + [image_output]
    )

    download_btn.click(fn=save_temp_image, inputs=image_output, outputs=download_btn)
