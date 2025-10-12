import gradio as gr
from gradio import themes
from PIL import Image, ImageEnhance, ImageFilter

# Global variable to store the original image

original_image = None


def apply_grayscale(img):
    return img.convert("L").convert("RGB")


def adjust_brightness(img, brightness):
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(brightness)


def adjust_contrast(img, contrast):
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(contrast)


def apply_blur(img, blur):
    return img.filter(ImageFilter.GaussianBlur(radius=blur))


def apply_sharpen(img, sharpen):
    for _ in range(int(sharpen)):
        img = img.filter(ImageFilter.SHARPEN)
    return img


def flip_image(img, flip_h, flip_v):
    if flip_h:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if flip_v:
        img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    return img


def rotate_image(img, rotation):
    return img.rotate(rotation, expand=True, fillcolor="white")


def process_image(
    image,
    grayscale,
    brightness,
    contrast,
    rotation,
    blur,
    sharpen,
    flip_h,
    flip_v,
):
    """Apply various image transformations based on user inputs."""
    global original_image

    if image is None:
        return None

    if original_image is None:
        original_image = image.copy()

    img = image.copy()

    if grayscale:
        img = apply_grayscale(img)

    if brightness != 1.0:
        img = adjust_brightness(img, brightness)

    if contrast != 1.0:
        img = adjust_contrast(img, contrast)

    if blur > 0:
        img = apply_blur(img, blur)

    if sharpen > 0:
        img = apply_sharpen(img, sharpen)

    img = flip_image(img, flip_h, flip_v)

    if rotation != 0:
        img = rotate_image(img, rotation)

    return img


def reset_image(original):
    """Reset all controls to default values and return original image."""

    global original_image

    if original_image is not None:
        return (
            original_image,  # output image
            False,  # grayscale
            1.0,  # brightness
            1.0,  # contrast
            0,  # rotation
            0,  # blur
            0,  # sharpen
            False,  # flip_h
            False,  # flip_v
        )

    return None, False, 1.0, 1.0, 0, 0, 0, False, False


def clear_all():
    """Clear everything including the original image."""

    global original_image

    original_image = None

    return None, None, False, 1.0, 1.0, 0, 0, 0, False, False


def on_image_upload(image):
    """Handle new image upload."""

    global original_image

    original_image = image.copy() if image is not None else None

    return image


# Create Gradio interface

with gr.Blocks(title="Image Editor Pro", theme=themes.Soft()) as imagedit_vibe_app:
    gr.Markdown(
        """
       # 🎨 Image Editor Pro
       Upload an image and apply various editing operations in real-time!
       """
    )

    with gr.Row():
        with gr.Column(scale=1):
            # Input section

            input_image = gr.Image(label="Upload Image", type="pil", sources=["upload", "clipboard"])

            gr.Markdown("### 🎛️ Basic Controls")

            grayscale_check = gr.Checkbox(label="Convert to Grayscale", value=False)

            brightness_slider = gr.Slider(minimum=0.5, maximum=1.5, value=1.0, step=0.1, label="Brightness")

            contrast_slider = gr.Slider(minimum=0.5, maximum=1.5, value=1.0, step=0.1, label="Contrast")

            rotation_slider = gr.Slider(minimum=-180, maximum=180, value=0, step=15, label="Rotation (degrees)")

            gr.Markdown("### ✨ Extra Effects")

            blur_slider = gr.Slider(minimum=0, maximum=10, value=0, step=1, label="Blur Intensity")

            sharpen_slider = gr.Slider(minimum=0, maximum=5, value=0, step=1, label="Sharpen Intensity")

            with gr.Row():
                flip_h_check = gr.Checkbox(label="Flip Horizontal", value=False)

                flip_v_check = gr.Checkbox(label="Flip Vertical", value=False)

            gr.Markdown("### 🔧 Actions")

            with gr.Row():
                reset_btn = gr.Button("🔄 Reset", variant="secondary")

                clear_btn = gr.Button("🗑️ Clear All", variant="stop")

        with gr.Column(scale=1):
            # Output section

            output_image = gr.Image(label="Edited Image", type="pil")

            download_btn = gr.Button("💾 Download Image", variant="primary", size="lg")

    # Event handlers

    inputs = [
        input_image,
        grayscale_check,
        brightness_slider,
        contrast_slider,
        rotation_slider,
        blur_slider,
        sharpen_slider,
        flip_h_check,
        flip_v_check,
    ]

    # Update image when upload happens

    input_image.change(fn=on_image_upload, inputs=[input_image], outputs=[output_image])

    # Update output when any control changes

    for component in inputs[1:]:  # Skip input_image
        component.change(fn=process_image, inputs=inputs, outputs=[output_image])

    # Reset button

    reset_btn.click(
        fn=reset_image,
        inputs=[input_image],
        outputs=[
            output_image,
            grayscale_check,
            brightness_slider,
            contrast_slider,
            rotation_slider,
            blur_slider,
            sharpen_slider,
            flip_h_check,
            flip_v_check,
        ],
    )

    # Clear all button

    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[
            input_image,
            output_image,
            grayscale_check,
            brightness_slider,
            contrast_slider,
            rotation_slider,
            blur_slider,
            sharpen_slider,
            flip_h_check,
            flip_v_check,
        ],
    )

    # Download functionality (Gradio handles this automatically with the output image)

    gr.Markdown(
        """

       ---

       💡 **Tips:**

       - Upload an image to get started

       - Adjust sliders and checkboxes to see real-time changes

       - Use Reset to revert to original while keeping your image

       - Use Clear All to start fresh with a new image

       - Right-click the edited image to download it

       """
    )


if __name__ == "__main__":
    imagedit_vibe_app.launch()
