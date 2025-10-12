import gradio as gr
from PIL import ImageEnhance


def convert_to_grayscale(image, grayscale_checkbox):
    if grayscale_checkbox:
        return image.convert("L")
    return image


def adjust_image_brightness(image, intensity):
    enhancer_brightness = ImageEnhance.Brightness(image)
    image = enhancer_brightness.enhance(intensity)
    return image


def rotate_image(image, angle):
    image = image.rotate(angle)
    return image


def adjust_image_contrast(image, intensity):
    enhancer_contrast = ImageEnhance.Contrast(image)
    image = enhancer_contrast.enhance(intensity)
    return image


def process_image(image, checkbox, brightness, angle, contrast):
    image = convert_to_grayscale(image, checkbox)
    image = adjust_image_brightness(image, brightness)
    image = adjust_image_contrast(image, contrast)
    image = rotate_image(image, angle)
    return image


def reset():
    return None, False, 1, 1, 0, None


with gr.Blocks() as image_app:
    gr.Markdown("# Image Processing App")

    with gr.Row():
        with gr.Column():
            grayscale_box = gr.Checkbox(label="Convert Image to Grayscale")
            brightness_slider = gr.Slider(0.5, 1.5, value=1, label="Brightness")
            contrast_slider = gr.Slider(0.5, 1.5, value=1, label="Contrast")
            rotation_slider = gr.Slider(-180, 180, value=0, label="Rotation Angle")
            output_button = gr.Button("Update Output")
            reset_button = gr.Button("Reset")

        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image")
            output_image = gr.Image(type="pil", label="Output Image", show_download_button=True)

    input_image.upload(fn=lambda x: x, inputs=input_image, outputs=output_image)

    output_button.click(fn=process_image,
                        inputs=[input_image, grayscale_box, brightness_slider,
                                rotation_slider, contrast_slider],
                        outputs=[output_image])
    reset_button.click(fn=reset, inputs=None,
                       outputs=[output_image, grayscale_box, brightness_slider,
                                contrast_slider, rotation_slider, input_image])
