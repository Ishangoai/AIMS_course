import gradio as gr
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def apply_sepia_filter(image):
    """Apply sepia filter to the image."""
    img_array = np.array(image)
    sepia_filter = np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]])
    sepia_img = img_array @ sepia_filter.T
    sepia_img = np.clip(sepia_img, 0, 255)
    return Image.fromarray(sepia_img.astype(np.uint8))


def apply_reverse_colors(image):
    """Apply reverse/invert colors to the image."""
    img_array = np.array(image)
    inverted_array = 255 - img_array
    return Image.fromarray(inverted_array.astype(np.uint8))


def apply_pil_filters(image, filter_options):
    """Apply PIL ImageFilter operations based on filter options."""
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


def apply_basic_adjustments(image, grayscale, brightness, contrast, rotation):
    """Apply basic image adjustments: grayscale, brightness, contrast, rotation."""
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


def manipulate_image(
    image,
    grayscale,
    brightness,
    contrast,
    rotation,
    sepia,
    reverse_colors,
    edge_contrast,
    blur,
    contour,
    detail,
    edge_enhance,
    edge_enhance_more,
    emboss,
    sharpen,
    smooth,
    smooth_more,
):
    """
    Apply various image manipulation operations to the input image.

    Args:
        image: PIL Image object
        grayscale: Boolean to convert to grayscale
        brightness: Float value for brightness adjustment (0.5 to 1.5)
        contrast: Float value for contrast adjustment (0.5 to 1.5)
        rotation: Integer value for rotation in degrees (-180 to 180)
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

    # Apply color effects
    if sepia:
        result = apply_sepia_filter(result)

    if reverse_colors:
        result = apply_reverse_colors(result)

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

    return result


with gr.Blocks(title="Image Manipulation App") as app:
    gr.Markdown("# 🎨 Image Manipulation App")
    gr.Markdown("Upload an image and apply various editing operations to see the results in real-time!")

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Upload Image", height=300)
            gr.Markdown("### 🛠️ Image Controls")
            grayscale_check = gr.Checkbox(label="Convert to Grayscale", value=False)
            brightness_slider = gr.Slider(
                minimum=0.5, maximum=1.5, value=1.0, step=0.1, label="Brightness", info="0.5 = Darker, 1.5 = Brighter"
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

            gr.Markdown("### 🎨 Filter Effects")
            sepia_check = gr.Checkbox(label="Sepia Filter", value=False)
            reverse_colors_check = gr.Checkbox(label="Reverse Colors", value=False)
            edge_contrast_check = gr.Checkbox(label="Edge Contrast", value=False)

            gr.Markdown("### 🔧 PIL ImageFilter Effects")
            blur_check = gr.Checkbox(label="Blur", value=False)
            contour_check = gr.Checkbox(label="Contour", value=False)
            detail_check = gr.Checkbox(label="Detail", value=False)
            edge_enhance_check = gr.Checkbox(label="Edge Enhance", value=False)
            edge_enhance_more_check = gr.Checkbox(label="Edge Enhance More", value=False)
            emboss_check = gr.Checkbox(label="Emboss", value=False)
            sharpen_check = gr.Checkbox(label="Sharpen", value=False)
            smooth_check = gr.Checkbox(label="Smooth", value=False)
            smooth_more_check = gr.Checkbox(label="Smooth More", value=False)

            reset_btn = gr.Button("🔄 Reset All", variant="secondary")

        with gr.Column(scale=1):
            output_image = gr.Image(type="pil", label="Processed Image", height=400)

    def reset_controls():
        return False, 1.0, 1.0, 0, False, False, False, False, False, False, False, False, False, False, False, False

    controls = [
        input_image,
        grayscale_check,
        brightness_slider,
        contrast_slider,
        rotation_slider,
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

    for control in controls:
        control.change(fn=manipulate_image, inputs=controls, outputs=output_image)

    reset_btn.click(
        fn=reset_controls,
        outputs=[
            grayscale_check,
            brightness_slider,
            contrast_slider,
            rotation_slider,
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
