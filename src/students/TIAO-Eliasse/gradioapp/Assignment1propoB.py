import os

import gradio as gr
from PIL import Image, ImageEnhance

# === Configuration ===
os.environ["GRADIO_TEMP_DIR"] = "/home/eliasse/Desktop/tmp_gradio"
os.environ["GRADIO_DISABLE_VIBE"] = "1"


def load_image(image):
    """Load the uploaded image into a Pillow Image object (RGB mode)."""
    if image is None:
        return None
    return image.convert("RGB")


def apply_grayscale(image, grayscale):
    """Apply grayscale conversion using Pillow if selected."""
    if grayscale:
        image = image.convert("L")
        image = image.convert("RGB")
    return image


def adjust_brightness(image, factor):
    """Adjust image brightness using Pillow's ImageEnhance."""
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def adjust_contrast(image, factor):
    """Adjust image contrast using Pillow's ImageEnhance."""
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def rotate_image(image, angle):
    """Rotate the image by a specified angle using Pillow."""
    return image.rotate(angle, expand=True)


def process_image(image, grayscale, brightness, contrast, rotation):
    """Apply all selected transformations using Pillow."""
    if image is None:
        return None

    img = load_image(image)
    img = apply_grayscale(img, grayscale)
    img = adjust_brightness(img, brightness)
    img = adjust_contrast(img, contrast)
    img = rotate_image(img, rotation)

    return img


def reset_image():
    """Reset all image transformation controls to default."""
    return False, 1.0, 1.0, 0


def save_image(image, format_choice):
    """
    Prepare the image for download in the selected format.
    Browser will prompt user where to save.

    Args:
        image: The image to save
        format_choice: Selected format (PNG or JPEG)

    Returns:
        str: Temporary file path for download
    """
    if image is None:
        return None

    # Convert NumPy array to PIL Image if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    # Create temp directory if it doesn't exist
    temp_dir = os.environ.get("GRADIO_TEMP_DIR", "/tmp")
    os.makedirs(temp_dir, exist_ok=True)

    # Determine filename and path
    if format_choice == "JPEG":
        filename = os.path.join(temp_dir, "edited_image.jpg")
        # JPEG doesn't support transparency, convert to RGB
        if image.mode in ("RGBA", "LA", "P"):
            rgb_image = Image.new("RGB", image.size, (255, 255, 255))
            if image.mode == "P":
                image = image.convert("RGBA")
            rgb_image.paste(image, mask=image.split()[-1] if image.mode in ("RGBA", "LA") else None)
            image = rgb_image
        image.save(filename, format="JPEG", quality=95)
    else:  # PNG
        filename = os.path.join(temp_dir, "edited_image.png")
        image.save(filename, format="PNG")

    return filename


# === Interface Gradio ===
with gr.Blocks(
    css="""
    body {background: #f7f9fc;}
    .input-section {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .output-section {
        background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    """,
    analytics_enabled=False
) as gradioImage:
    gr.Markdown("## Pillow Image Editing App\nUpload and enhance your image easily!")

    with gr.Row():
        with gr.Column(elem_classes="input-section"):
            input_img = gr.Image(type="pil", label="Upload Image")
            grayscale = gr.Checkbox(label="Convert to Grayscale", value=False)
            brightness = gr.Slider(0.5, 1.5, value=1.0, label="Brightness")
            contrast = gr.Slider(0.5, 1.5, value=1.0, label="Contrast")
            rotation = gr.Slider(-180, 180, value=0, label="Rotate (°)")
            reset_btn = gr.Button("Reset")

        with gr.Column(elem_classes="output-section"):
            output_img = gr.Image(label="Edited Image")
            format_dropdown = gr.Dropdown(
                choices=["PNG", "JPEG"],
                value="PNG",
                label="Select Download Format"
            )
            download_btn = gr.Button("Download")

    # Connect controls
    controls = [input_img, grayscale, brightness, contrast, rotation]
    for ctrl in controls:
        ctrl.change(process_image, inputs=controls, outputs=output_img)

    reset_btn.click(fn=reset_image, outputs=[grayscale, brightness, contrast, rotation])
    download_btn.click(
        fn=save_image,
        inputs=[output_img, format_dropdown],
        outputs=gr.File(label="Download Edited Image")
    )


if __name__ == "__main__":
    gradioImage.launch()
