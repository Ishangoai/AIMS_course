import gradio as gr
from PIL import ImageEnhance


def edit_image(img, brightness, contrast, apply_grayscale, rotate_angle):
    image = img.copy()
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)
    image = image.rotate(rotate_angle)

    if apply_grayscale:
        image = image.convert("L").convert("RGB")

    return image


with gr.Blocks(css="body {background: #f2f7ff;}") as image_edit_v2_app:
    gr.Markdown('# Imagedit 🖼️')
    gr.Markdown('Choose a picture you want to edit below')

    with gr.Row():
        # Input Column 1
        with gr.Column():
            input_image = gr.Image(type='pil')
        # Image input
        with gr.Column():
            # gr.Markdown('# Edited Image')
            output_image = gr.Image(format='png')
            gr.Button('Clear')
            gr.Button('Download')
        #     pass

        with gr.Column():
            apply_grayscale = gr.Checkbox(value=False, label='Convert Image to Grayscale')
            brightness = gr.Slider(minimum=0.5, value=1.0, maximum=1.5, label='Adjust Brightness')
            contrast = gr.Slider(minimum=0.5, value=1.0, maximum=1.5, label='Adjust Contrast')
            rotate_angle = gr.Slider(minimum=-180, value=0, maximum=180, label='Rotate Image')

        print("-----------------------------------------------------")
        print(f"Image Typpe: {type(input_image), type(input_image.value)}", flush=True)
        print()
        print()
        print("-----------------------------------------------------")
        inputs = [input_image, brightness, contrast, apply_grayscale, rotate_angle]

        if input_image.value is not None:
            for component in inputs:
                # processed_image = edit_image(
                #     input_image, brightness, contrast, 
                #     apply_grayscale=False, rotate_angle=rotate_angle
                #     )
                component.change(fn=edit_image, inputs=inputs, outputs=output_image)
