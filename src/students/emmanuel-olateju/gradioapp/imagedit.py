import gradio as gr
from gradioapp.utils.heart_disease_utils import predict_heart_disease
from PIL import ImageEnhance


def wrapped_predict(*args):
    return predict_heart_disease(list(args))


def edit_image(
    img,
    brightness,
    contrast,
    apply_grayscale,
):
    image = img.copy()
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)

    if apply_grayscale:
        image = image.convert("L").convert("RGB")

    return image


with gr.Blocks(css="body {background: #f2f7ff;}") as image_edit_app:
    gr.Markdown('# Imagedit 🖼️')
    gr.Markdown('Choose a picture you want to edit below')

    with gr.Row():
        # Input Column 1
        with gr.Column():
            input_image = gr.Image(type='pil')

        # Image input
        with gr.Column():
            # gr.Markdown('# Edited Image')
            output_image = gr.Image()
            gr.Button('Clear')
            gr.Button('Download')
        #     pass

        with gr.Column():
            apply_grayscale = gr.Checkbox(value=False, label='Convert Image to Grayscale')
            brightness = gr.Slider(minimum=0.5, value=1.0, maximum=1.5, label='Adjust Brightness')
            contrast = gr.Slider(minimum=0.5, value=1.0, maximum=1.5, label='Adjust Contrast')

        inputs = [input_image, brightness, contrast, apply_grayscale]

        for component in inputs:
            component.change(fn=edit_image, inputs=inputs, outputs=output_image)
