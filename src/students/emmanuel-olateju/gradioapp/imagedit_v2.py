import gradio as gr
from PIL import Image, ImageEnhance

original_image = None
processed_image = None


def upload_image(file):
    """Upload a new image from the filesystem."""
    if file is None:
        return None
    return Image.open(file)


def on_image_upload(image):
    """Action triggered when an image is uploaded"""
    global original_image, processed_image
    original_image = image.copy() if image is not None else None
    processed_image = image.copy() if image is not None else None
    return image


def show_upload():
    """Show the image upload component"""
    return gr.update(visible=True)


def clear_image():
    """clear all the current image in the canvas."""
    global original_image, processed_image
    original_image = None
    processed_image = None
    return None, None, 1.0, 1.0, False, 0.0


def reset_image(image):
    global original_image, processed_image
    if original_image is not None:
        processed_image = original_image.copy()
        return original_image, 1.0, 1.0, False, 0.0
    return None, 1.0, 1.0, False, 0.0


def edit_image(img, brightness, contrast, apply_grayscale, rotate_angle):
    global processed_image
    if img is None:
        return None
    image = img.copy()
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)
    image = image.rotate(rotate_angle, expand=True)
    if apply_grayscale:
        image = image.convert("L").convert("RGB")

    # Store the processed image globally
    processed_image = image.copy()
    return image


def download_image():
    """Prepare image for download - uses the globally stored processed image"""
    global processed_image
    if processed_image is None:
        return None

    # Save the processed PIL image to a temporary file
    temp_path = "edited_image.jpg"
    processed_image.save(temp_path, format="JPEG", quality=95)
    return temp_path


with gr.Blocks(css="body {background: #f2f7ff;}") as image_edit_v2_app:
    gr.Markdown("# Imagedit 🖼️")
    gr.Markdown("Choose a picture you want to edit below")

    with gr.Row():
        # Input Column 1
        with gr.Column():
            # upload_button = gr.Button("Upload image")
            input_image = gr.Image(type="pil", label="Upload Image")
            apply_grayscale = gr.Checkbox(value=False, label="Convert Image to Grayscale")
            brightness = gr.Slider(minimum=0.5, value=1.0, maximum=1.5, label="Adjust Brightness")
            contrast = gr.Slider(minimum=0.5, value=1.0, maximum=1.5, label="Adjust Contrast")
            rotate_angle = gr.Slider(minimum=-180, value=0, maximum=180, label="Rotate Image")
            clear_buttn = gr.Button("Clear")

        # Image output
        with gr.Column(scale=3):
            output_image = gr.Image(format="png", label="Edited Image")
            reset_buttn = gr.Button("Reset")
            download_button = gr.DownloadButton("Download", variant="primary")

    inputs = [input_image, brightness, contrast, apply_grayscale, rotate_angle]

    # When image is uploaded, copy to output and save original
    input_image.change(fn=on_image_upload, inputs=[input_image], outputs=[output_image])

    # When any slider/checkbox changes, apply edits
    for component in inputs[1:]:
        component.change(fn=edit_image, inputs=inputs, outputs=output_image)

    # Reset button: restore original image and reset all controls
    reset_buttn.click(
        fn=reset_image,
        inputs=input_image,
        outputs=[
            output_image,
            brightness,
            contrast,
            apply_grayscale,
            rotate_angle,
        ],
    )

    # Clear button: clear everything
    clear_buttn.click(
        fn=clear_image,
        inputs=[],
        outputs=[
            input_image,
            output_image,
            brightness,
            contrast,
            apply_grayscale,
            rotate_angle,
        ],
    )

    # Download button: download the edited image
    download_button.click(fn=download_image, inputs=[], outputs=download_button)

if __name__ == "__main__":
    image_edit_v2_app.launch()
