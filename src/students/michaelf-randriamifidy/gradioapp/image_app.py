import gradio as gr
from gradioapp.utils.image_app_utils import edit_image

with gr.Blocks(css="body {background: #f2f7ff;}") as image_transformation:
    gr.Markdown("# Image Transformation")
    gr.Markdown("Edit uploaded image")

    with gr.Row():
        with gr.Column():
            grayscale = gr.Checkbox(label="Grayscale", info="Grayscale?")
            brightness = gr.Slider(0, 100, value=0, label="Brightness")
            contrast = gr.Slider(0, 100, value=0, label="Contrast")
            rotate_angle = gr.Slider(0, 360, value=0, label="Contrast")
    edit_btn = gr.Button("Edit")
    result = gr.Textbox(label="Result")
    edit_btn.click(fn=edit_image,
                inputs=[grayscale, brightness, contrast, rotate_angle],
                outputs=result)
# image_app = gr.Interface(
#     fn=to_grayscale,
#     inputs=gr.Image(type="pil"),
#     outputs=gr.Image(type="pil"),
#     title="Grayscale Image Converter",
#     description="Upload an image and see it converted to grayscale."
# )
