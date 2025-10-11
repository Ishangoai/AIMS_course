import gradio as gr
from PIL import ImageEnhance


def edit_image(img, grayscale, brightness, contrast, rotate):
    if img is None:
        return None
    image = img.copy()
    if grayscale:
        image = image.convert("L").convert("RGB")
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness)
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)
    if rotate != 0:
        image = image.rotate(rotate, expand=True)
    return image


with gr.Blocks() as demo:
    with gr.Row():
        input_img = gr.Image(label="Upload Image", type="pil")
        output_img = gr.Image(label="Edited Image")
    with gr.Row():
        grayscale = gr.Checkbox(label="Convert to Grayscale")
        brightness = gr.Slider(0.5, 1.5, value=1.0, label="Brightness")
        contrast = gr.Slider(0.5, 1.5, value=1.0, label="Contrast")
        rotate = gr.Slider(-180, 180, value=0, label="Rotate (degrees)")
    input_img.change(
        fn=edit_image,
        inputs=[input_img, grayscale, brightness, contrast, rotate],
        outputs=output_img
    )
    grayscale.change(
        fn=edit_image,
        inputs=[input_img, grayscale, brightness, contrast, rotate],
        outputs=output_img
    )
    brightness.change(
        fn=edit_image,
        inputs=[input_img, grayscale, brightness, contrast, rotate],
        outputs=output_img
    )
    contrast.change(
        fn=edit_image,
        inputs=[input_img, grayscale, brightness, contrast, rotate],
        outputs=output_img
    )
    rotate.change(
        fn=edit_image,
        inputs=[input_img, grayscale, brightness, contrast, rotate],
        outputs=output_img
    )

if __name__ == "__main__":
    demo.launch()
