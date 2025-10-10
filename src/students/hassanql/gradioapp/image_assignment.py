import gradio as gr
from PIL import Image, ImageEnhance
import numpy as np
import tempfile


def edit_image(image, grayscale, brightness, contrast, rotation):
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
        return image, False, 1.0, 1.0, 0, image

    reset_btn.click(
        fn=reset_to_original,
        inputs=[image_input],
        outputs=inputs + [image_output],
    )

    # Download button saves edited image to temp file
    download_btn.click(fn=save_temp_image, inputs=image_output, outputs=download_btn)
