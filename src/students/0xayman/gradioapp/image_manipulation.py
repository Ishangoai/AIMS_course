from __future__ import annotations

import gradio as gr
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def apply_sepia_filter(image: Image.Image) -> Image.Image:
    """
    Apply a sepia filter to the input image.

    Args:
        image (Image.Image): The input PIL Image object to apply sepia filter to.

    Returns:
        Image.Image: A new PIL Image object with sepia filter applied.
    """
    img_array = np.array(image)
    sepia_filter = np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]])
    sepia_img = img_array @ sepia_filter.T
    sepia_img = np.clip(sepia_img, 0, 255)
    return Image.fromarray(sepia_img.astype(np.uint8))


def apply_reverse_colors(image: Image.Image) -> Image.Image:
    """
    Apply reverse/invert colors to the input image.

    Args:
        image (Image.Image): The input PIL Image object to invert colors for.

    Returns:
        Image.Image: A new PIL Image object with inverted colors.
    """
    img_array = np.array(image)
    inverted_array = 255 - img_array
    return Image.fromarray(inverted_array.astype(np.uint8))


def apply_pil_filters(image: Image.Image, filter_options: dict[str, bool]) -> Image.Image:
    """
    Apply PIL ImageFilter operations based on filter options.

    Args:
        image (Image.Image): The input PIL Image object to apply filters to.
        filter_options (dict[str, bool]): Dictionary mapping filter names to boolean values
                                         indicating whether to apply each filter.

    Returns:
        Image.Image: A new PIL Image object with the specified filters applied.
    """
    result = image
    filter_map = {
        "edge_contrast": ImageFilter.FIND_EDGES,
        "blur": ImageFilter.BLUR,
        "contour": ImageFilter.CONTOUR,
        "detail": ImageFilter.DETAIL,
        "edge_enhance": ImageFilter.EDGE_ENHANCE,
        "edge_enhance_more": ImageFilter.EDGE_ENHANCE_MORE,
        "emboss": ImageFilter.EMBOSS,
        "sharpen": ImageFilter.SHARPEN,
        "smooth": ImageFilter.SMOOTH,
        "smooth_more": ImageFilter.SMOOTH_MORE,
    }

    for filter_name, should_apply in filter_options.items():
        if should_apply and filter_name in filter_map:
            result = result.filter(filter_map[filter_name])

    return result


def apply_basic_adjustments(
    image: Image.Image, grayscale: bool, brightness: float, contrast: float, rotation: int
) -> Image.Image:
    """
    Apply basic image adjustments: grayscale, brightness, contrast, rotation.

    Args:
        image (Image.Image): The input PIL Image object to adjust.
        grayscale (bool): Whether to convert the image to grayscale.
        brightness (float): Brightness adjustment factor (1.0 = no change, <1.0 = darker, >1.0 = brighter).
        contrast (float): Contrast adjustment factor (1.0 = no change, <1.0 = lower contrast, >1.0 = higher contrast).
        rotation (int): Rotation angle in degrees (positive = clockwise, negative = counterclockwise).

    Returns:
        Image.Image: A new PIL Image object with the specified adjustments applied.
    """
    result = image

    if grayscale:
        result = result.convert("L").convert("RGB")

    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(result)
        result = enhancer.enhance(brightness)

    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(result)
        result = enhancer.enhance(contrast)

    if rotation != 0:
        result = result.rotate(rotation, expand=True, fillcolor=(255, 255, 255))

    return result


def apply_advanced_adjustments(
    image: Image.Image, saturation: float, sharpness: float, gamma: float
) -> Image.Image:
    """
    Apply advanced image adjustments: saturation, sharpness, gamma correction.

    Args:
        image (Image.Image): The input PIL Image object to adjust.
        saturation (float): Saturation adjustment factor (1.0 = no change, 0.0 = grayscale, >1.0 = more saturated).
        sharpness (float): Sharpness adjustment factor (1.0 = no change, <1.0 = blurred, >1.0 = sharper).
        gamma (float): Gamma correction factor (1.0 = no change, <1.0 = darker, >1.0 = brighter).

    Returns:
        Image.Image: A new PIL Image object with the specified adjustments applied.
    """
    result = image

    if saturation != 1.0:
        enhancer = ImageEnhance.Color(result)
        result = enhancer.enhance(saturation)

    if sharpness != 1.0:
        enhancer = ImageEnhance.Sharpness(result)
        result = enhancer.enhance(sharpness)

    if gamma != 1.0:
        # Apply gamma correction
        img_array = np.array(result)
        gamma_corrected = np.power(img_array / 255.0, gamma) * 255.0
        gamma_corrected = np.clip(gamma_corrected, 0, 255)
        result = Image.fromarray(gamma_corrected.astype(np.uint8))

    return result


def apply_transformations(image: Image.Image, flip_horizontal: bool, flip_vertical: bool) -> Image.Image:
    """
    Apply geometric transformations: horizontal and vertical flipping.

    Args:
        image (Image.Image): The input PIL Image object to transform.
        flip_horizontal (bool): Whether to flip the image horizontally.
        flip_vertical (bool): Whether to flip the image vertically.

    Returns:
        Image.Image: A new PIL Image object with the specified transformations applied.
    """
    result = image

    if flip_horizontal:
        result = result.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    if flip_vertical:
        result = result.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    return result


def apply_color_channel_effects(
    image: Image.Image, red_only: bool, green_only: bool, blue_only: bool
) -> Image.Image:
    """
    Apply color channel isolation effects.

    Args:
        image (Image.Image): The input PIL Image object to process.
        red_only (bool): Whether to show only the red channel.
        green_only (bool): Whether to show only the green channel.
        blue_only (bool): Whether to show only the blue channel.

    Returns:
        Image.Image: A new PIL Image object with the specified channel effects applied.
    """
    if not any([red_only, green_only, blue_only]):
        return image

    img_array = np.array(image)
    result_array = img_array.copy()

    if red_only:
        result_array[:, :, 1] = 0  # Green channel
        result_array[:, :, 2] = 0  # Blue channel
    elif green_only:
        result_array[:, :, 0] = 0  # Red channel
        result_array[:, :, 2] = 0  # Blue channel
    elif blue_only:
        result_array[:, :, 0] = 0  # Red channel
        result_array[:, :, 1] = 0  # Green channel

    return Image.fromarray(result_array.astype(np.uint8))


def apply_special_effects(
    image: Image.Image, solarize: bool, posterize: bool, posterize_bits: int, histogram_equalize: bool
) -> Image.Image:
    """
    Apply special image effects: solarize, posterize, histogram equalization.

    Args:
        image (Image.Image): The input PIL Image object to process.
        solarize (bool): Whether to apply solarize effect.
        posterize (bool): Whether to apply posterize effect.
        posterize_bits (int): Number of bits for posterization (1-8).
        histogram_equalize (bool): Whether to apply histogram equalization.

    Returns:
        Image.Image: A new PIL Image object with the specified effects applied.
    """
    result = image

    if solarize:
        result = ImageOps.solarize(result, threshold=128)

    if posterize:
        result = ImageOps.posterize(result, posterize_bits)

    if histogram_equalize:
        result = ImageOps.equalize(result)

    return result


def resize_image(image: Image.Image, resize_enabled: bool, width: int, height: int) -> Image.Image:
    """
    Resize the image to specified dimensions.

    Args:
        image (Image.Image): The input PIL Image object to resize.
        resize_enabled (bool): Whether to apply resizing.
        width (int): Target width in pixels.
        height (int): Target height in pixels.

    Returns:
        Image.Image: A new PIL Image object with the specified size.
    """
    if not resize_enabled:
        return image

    return image.resize((width, height), Image.Resampling.LANCZOS)


def manipulate_image(
    image: Image.Image | None,
    grayscale: bool,
    brightness: float,
    contrast: float,
    rotation: int,
    saturation: float,
    sharpness: float,
    gamma: float,
    flip_horizontal: bool,
    flip_vertical: bool,
    resize_enabled: bool,
    resize_width: int,
    resize_height: int,
    red_only: bool,
    green_only: bool,
    blue_only: bool,
    solarize: bool,
    posterize: bool,
    posterize_bits: int,
    histogram_equalize: bool,
    sepia: bool,
    reverse_colors: bool,
    edge_contrast: bool,
    blur: bool,
    contour: bool,
    detail: bool,
    edge_enhance: bool,
    edge_enhance_more: bool,
    emboss: bool,
    sharpen: bool,
    smooth: bool,
    smooth_more: bool,
) -> Image.Image | None:
    """
    Apply various image manipulation operations to the input image.

    Args:
        image: PIL Image object
        grayscale: Boolean to convert to grayscale
        brightness: Float value for brightness adjustment (0.5 to 1.5)
        contrast: Float value for contrast adjustment (0.5 to 1.5)
        rotation: Integer value for rotation in degrees (-180 to 180)
        saturation: Float value for saturation adjustment (0.0 to 2.0)
        sharpness: Float value for sharpness adjustment (0.0 to 2.0)
        gamma: Float value for gamma correction (0.5 to 2.0)
        flip_horizontal: Boolean to flip image horizontally
        flip_vertical: Boolean to flip image vertically
        resize_enabled: Boolean to enable resizing
        resize_width: Integer width for resizing
        resize_height: Integer height for resizing
        red_only: Boolean to show only red channel
        green_only: Boolean to show only green channel
        blue_only: Boolean to show only blue channel
        solarize: Boolean to apply solarize effect
        posterize: Boolean to apply posterize effect
        posterize_bits: Integer bits for posterization (1-8)
        histogram_equalize: Boolean to apply histogram equalization
        sepia: Boolean to apply sepia filter
        reverse_colors: Boolean to reverse/invert colors
        edge_contrast: Boolean to apply edge detection/contrast filter
        blur: Boolean to apply blur filter
        contour: Boolean to apply contour filter
        detail: Boolean to apply detail enhancement filter
        edge_enhance: Boolean to apply edge enhancement filter
        edge_enhance_more: Boolean to apply strong edge enhancement filter
        emboss: Boolean to apply emboss filter
        sharpen: Boolean to apply sharpen filter
        smooth: Boolean to apply smooth filter
        smooth_more: Boolean to apply strong smooth filter

    Returns:
        PIL Image object with applied manipulations
    """
    if image is None:
        return None

    result = image.copy()

    # Apply resizing first if enabled
    result = resize_image(result, resize_enabled, resize_width, resize_height)

    # Apply geometric transformations
    result = apply_transformations(result, flip_horizontal, flip_vertical)

    # Apply color effects
    if sepia:
        result = apply_sepia_filter(result)

    if reverse_colors:
        result = apply_reverse_colors(result)

    # Apply color channel effects
    result = apply_color_channel_effects(result, red_only, green_only, blue_only)

    # Apply special effects
    result = apply_special_effects(result, solarize, posterize, posterize_bits, histogram_equalize)

    # Apply PIL filters
    filter_options = {
        "edge_contrast": edge_contrast,
        "blur": blur,
        "contour": contour,
        "detail": detail,
        "edge_enhance": edge_enhance,
        "edge_enhance_more": edge_enhance_more,
        "emboss": emboss,
        "sharpen": sharpen,
        "smooth": smooth,
        "smooth_more": smooth_more,
    }
    result = apply_pil_filters(result, filter_options)

    # Apply basic adjustments
    result = apply_basic_adjustments(result, grayscale, brightness, contrast, rotation)

    # Apply advanced adjustments
    result = apply_advanced_adjustments(result, saturation, sharpness, gamma)

    return result


with gr.Blocks(title="Image Manipulation App") as app:
    gr.Markdown("# 🎨 Image Manipulation App")
    gr.Markdown("Upload an image and apply various editing operations to see the results in real-time!")

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Upload Image", height=300)

            with gr.Tabs():
                with gr.TabItem("🔧 Basic Controls"):
                    grayscale_check = gr.Checkbox(label="Convert to Grayscale", value=False)
                    brightness_slider = gr.Slider(
                        minimum=0.5,
                        maximum=1.5,
                        value=1.0,
                        step=0.1,
                        label="Brightness",
                        info="0.5 = Darker, 1.5 = Brighter"
                    )
                    contrast_slider = gr.Slider(
                        minimum=0.5,
                        maximum=1.5,
                        value=1.0,
                        step=0.1,
                        label="Contrast",
                        info="0.5 = Lower contrast, 1.5 = Higher contrast",
                    )
                    rotation_slider = gr.Slider(
                        minimum=-180, maximum=180, value=0, step=15, label="Rotation (degrees)", info="-180° to 180°"
                    )

                with gr.TabItem("✨ Advanced Controls"):
                    saturation_slider = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Saturation",
                        info="0.0 = Grayscale, 2.0 = Very saturated"
                    )
                    sharpness_slider = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Sharpness",
                        info="0.0 = Blurred, 2.0 = Very sharp"
                    )
                    gamma_slider = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Gamma Correction",
                        info="0.5 = Darker, 2.0 = Brighter"
                    )

                with gr.TabItem("↕️ Transform"):
                    flip_horizontal_check = gr.Checkbox(label="Flip Horizontal", value=False)
                    flip_vertical_check = gr.Checkbox(label="Flip Vertical", value=False)

                    gr.Markdown("### 📏 Resize")
                    resize_enabled_check = gr.Checkbox(label="Enable Resize", value=False)
                    resize_width_slider = gr.Slider(
                        minimum=50, maximum=2000, value=512, step=10, label="Width (pixels)"
                    )
                    resize_height_slider = gr.Slider(
                        minimum=50, maximum=2000, value=512, step=10, label="Height (pixels)"
                    )

                with gr.TabItem("🌈 Color Effects"):
                    sepia_check = gr.Checkbox(label="Sepia Filter", value=False)
                    reverse_colors_check = gr.Checkbox(label="Reverse Colors", value=False)

                    gr.Markdown("### RGB Channel Isolation")
                    red_only_check = gr.Checkbox(label="Red Channel Only", value=False)
                    green_only_check = gr.Checkbox(label="Green Channel Only", value=False)
                    blue_only_check = gr.Checkbox(label="Blue Channel Only", value=False)

                with gr.TabItem("🎭 Special Effects"):
                    solarize_check = gr.Checkbox(label="Solarize", value=False)
                    posterize_check = gr.Checkbox(label="Posterize", value=False)
                    posterize_bits_slider = gr.Slider(
                        minimum=1, maximum=8, value=4, step=1, label="Posterize Bits", info="Lower = fewer colors"
                    )
                    histogram_equalize_check = gr.Checkbox(label="Histogram Equalization", value=False)

                with gr.TabItem("� PIL Filters"):
                    edge_contrast_check = gr.Checkbox(label="Edge Contrast", value=False)
                    blur_check = gr.Checkbox(label="Blur", value=False)
                    contour_check = gr.Checkbox(label="Contour", value=False)
                    detail_check = gr.Checkbox(label="Detail", value=False)
                    edge_enhance_check = gr.Checkbox(label="Edge Enhance", value=False)
                    edge_enhance_more_check = gr.Checkbox(label="Edge Enhance More", value=False)
                    emboss_check = gr.Checkbox(label="Emboss", value=False)
                    sharpen_check = gr.Checkbox(label="Sharpen", value=False)
                    smooth_check = gr.Checkbox(label="Smooth", value=False)
                    smooth_more_check = gr.Checkbox(label="Smooth More", value=False)

            reset_btn = gr.Button("🔄 Reset All", variant="secondary", size="lg")

        with gr.Column(scale=1):
            output_image = gr.Image(type="pil", label="Processed Image", height=500)

            with gr.Row():
                gr.Markdown("### 📊 Image Info")
            info_text = gr.Textbox(label="Image Information", lines=3, interactive=False)

    def get_image_info(image: Image.Image | None) -> str:
        """
        Get information about the image dimensions and format.

        Args:
            image (Image.Image | None): The PIL Image object to analyze.

        Returns:
            str: Formatted string with image information.
        """
        if image is None:
            return "No image loaded"

        width, height = image.size
        mode = image.mode
        format_name = getattr(image, 'format', 'Unknown')

        return f"Size: {width} x {height} pixels\nMode: {mode}\nFormat: {format_name}"

    def reset_controls() -> tuple:
        """
        Reset all image manipulation controls to their default values.

        Returns:
            tuple: A tuple containing default values for all controls.
        """
        return (
            False,  # grayscale
            1.0,    # brightness
            1.0,    # contrast
            0,      # rotation
            1.0,    # saturation
            1.0,    # sharpness
            1.0,    # gamma
            False,  # flip_horizontal
            False,  # flip_vertical
            False,  # resize_enabled
            512,    # resize_width
            512,    # resize_height
            False,  # red_only
            False,  # green_only
            False,  # blue_only
            False,  # solarize
            False,  # posterize
            4,      # posterize_bits
            False,  # histogram_equalize
            False,  # sepia
            False,  # reverse_colors
            False,  # edge_contrast
            False,  # blur
            False,  # contour
            False,  # detail
            False,  # edge_enhance
            False,  # edge_enhance_more
            False,  # emboss
            False,  # sharpen
            False,  # smooth
            False,  # smooth_more
        )

    controls = [
        input_image,
        grayscale_check,
        brightness_slider,
        contrast_slider,
        rotation_slider,
        saturation_slider,
        sharpness_slider,
        gamma_slider,
        flip_horizontal_check,
        flip_vertical_check,
        resize_enabled_check,
        resize_width_slider,
        resize_height_slider,
        red_only_check,
        green_only_check,
        blue_only_check,
        solarize_check,
        posterize_check,
        posterize_bits_slider,
        histogram_equalize_check,
        sepia_check,
        reverse_colors_check,
        edge_contrast_check,
        blur_check,
        contour_check,
        detail_check,
        edge_enhance_check,
        edge_enhance_more_check,
        emboss_check,
        sharpen_check,
        smooth_check,
        smooth_more_check,
    ]

    # Handle image manipulation
    for control in controls:
        control.change(fn=manipulate_image, inputs=controls, outputs=output_image)

    # Handle image info display
    input_image.change(fn=get_image_info, inputs=input_image, outputs=info_text)

    # Handle reset button
    reset_btn.click(
        fn=reset_controls,
        outputs=[
            grayscale_check,
            brightness_slider,
            contrast_slider,
            rotation_slider,
            saturation_slider,
            sharpness_slider,
            gamma_slider,
            flip_horizontal_check,
            flip_vertical_check,
            resize_enabled_check,
            resize_width_slider,
            resize_height_slider,
            red_only_check,
            green_only_check,
            blue_only_check,
            solarize_check,
            posterize_check,
            posterize_bits_slider,
            histogram_equalize_check,
            sepia_check,
            reverse_colors_check,
            edge_contrast_check,
            blur_check,
            contour_check,
            detail_check,
            edge_enhance_check,
            edge_enhance_more_check,
            emboss_check,
            sharpen_check,
            smooth_check,
            smooth_more_check,
        ],
    )

if __name__ == "__main__":
    app.launch()
