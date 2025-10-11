import gradio as gr
from PIL import Image, ImageEnhance, ImageOps

# The core function to edit the image
def edit_image(image, grayscale, brightness, contrast, rotation):
    if image is None:
        return None, None  # Return None for both image and a downloadable file

    try:
        # Convert numpy array from Gradio to a PIL Image
        edited_image = Image.fromarray(image)

        # Apply grayscale if checked
        if grayscale:
            edited_image = ImageOps.grayscale(edited_image)

        # Adjust brightness
        enhancer_brightness = ImageEnhance.Brightness(edited_image)
        edited_image = enhancer_brightness.enhance(brightness)

        # Adjust contrast
        enhancer_contrast = ImageEnhance.Contrast(edited_image)
        edited_image = enhancer_contrast.enhance(contrast)

        # Rotate the image
        if rotation != 0:
            edited_image = edited_image.rotate(rotation, expand=True, fillcolor='white')

        return edited_image, edited_image
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None


# Function to reset all controls to their default values
def reset_all():
    return None, False, 1.0, 1.0, 0, None, None

# Building the Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as image_app:
    gr.Markdown("# Simple Image Editor")
    gr.Markdown("Upload an image and use the controls to edit it. The edited image will be displayed on the right.")

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="numpy", label="Input Image")
            grayscale_check = gr.Checkbox(label="Convert to Grayscale")
            brightness_slider = gr.Slider(minimum=0.5, maximum=1.5, value=1.0, label="Brightness")
            contrast_slider = gr.Slider(minimum=0.5, maximum=1.5, value=1.0, label="Contrast")
            rotation_slider = gr.Slider(minimum=-180, maximum=180, value=0, label="Rotation (degrees)")

            with gr.Row():
                reset_btn = gr.Button("Reset")
                # The download button is part of the gr.File component

        with gr.Column(scale=2):
            output_image = gr.Image(type="pil", label="Output Image")
            download_file = gr.File(label="Download Edited Image", visible=True)

    # Define the components that will act as inputs to the edit_image function
    inputs = [input_image, grayscale_check, brightness_slider, contrast_slider, rotation_slider]

    # When any input component changes, call the edit_image function
    for component in inputs:
        component.change(fn=edit_image, inputs=inputs, outputs=[output_image, download_file])

    # Define what happens when the reset button is clicked
    reset_btn.click(
        fn=reset_all,
        inputs=[],
        outputs=[input_image, grayscale_check, brightness_slider, contrast_slider, rotation_slider, output_image, download_file]
    )

# Launch the application
# demo.launch()