import os
import tempfile

import gradio as gr
from backgroundremover.bg import remove
from PIL import Image


def remove_bg(image):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as input_file:
        image.save(input_file.name)
        input_file.close()
        with open(input_file.name, "rb") as f:
            input_bytes = f.read()
        output_bytes = remove(input_bytes)
        output_path = input_file.name.replace(".png", "_out.png")
        with open(output_path, "wb") as f:
            f.write(output_bytes)
        result = Image.open(output_path).convert("RGBA")
        os.remove(input_file.name)
        os.remove(output_path)
        return result


app = gr.Interface(
    fn=remove_bg,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="Background Remover",
    description="Upload an image to remove its background using backgroundremover."
)

if __name__ == "__main__":
    app.launch()
