import gradio as gr
from PIL import Image, ImageEnhance


def edit_image(image, grayscale, brightness, contrast, rotation):
    if image is None:
        return None

    img = image.convert("RGB")

    if grayscale:
        img = img.convert("L").convert("RGB")

    enhancer_b = ImageEnhance.Brightness(img)
    img = enhancer_b.enhance(brightness)

    enhancer_c = ImageEnhance.Contrast(img)
    img = enhancer_c.enhance(contrast)

    img = img.rotate(rotation, expand=True)

    return img


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
        with gr.Column():
            image_output = gr.Image(label="Edited Image")

    inputs = [image_input, grayscale, brightness, contrast, rotation]

    # Automatically update on input changes
    for inp in inputs:
        inp.change(fn=edit_image, inputs=inputs, outputs=image_output)

