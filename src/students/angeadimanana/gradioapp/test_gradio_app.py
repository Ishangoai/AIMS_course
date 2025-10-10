import gradio as gr

import requests

API_URL = "http://0.0.0.0:8080"


def to_grayscale(img):
    gray = img.convert("L")
    return gray

appli = gr.Interface(
    fn=to_grayscale,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="Grayscale Image Converter",
    description="Upload an image and see it converted to grayscale"
)

appli.launch()