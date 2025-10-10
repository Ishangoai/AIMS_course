import gradio as gr
from gradioapp.utils.image_app_utils import edit_image
from PIL import Image, ImageEnhance
import tempfile, os

def combined_effects(img, brightness, contrast, angle, grayscale):
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = img.rotate(angle, expand=True)
    if grayscale:
        img = img.convert("L")
    return img

# --- Fonction Reset ---
def reset_image(original_img):
    """Retourne simplement l'image originale."""
    return original_img

def save_image(img):
    temp_dir = tempfile.mkdtemp()
    save_path = os.path.join(temp_dir,"modified_image.png")
    img.save(save_path)
    return save_path
#with gr.Blocks(css="body {background: #f2f7ff;}") as image_transformation:
    #gr.Markdown("# Image Transformation")
    #gr.Markdown("Edit uploaded image")

    #with gr.Row():
        #with gr.Column():
            #grayscale = gr.Checkbox(label="Grayscale", info="Grayscale?")
            #brightness = gr.Slider(0, 100, value=0, label="Brightness")
            #contrast = gr.Slider(0, 100, value=0, label="Contrast")
            #rotate_angle = gr.Slider(0, 360, value=0, label="Contrast")
    #edit_btn = gr.Button("Edit")
    #result = gr.Textbox(label="Result")
    #edit_btn.click(fn=edit_image,
                #inputs=[grayscale, brightness, contrast, rotate_angle],
                #outputs=result)
# image_app = gr.Interface(
#     fn=to_grayscale,
#     inputs=gr.Image(type="pil"),
#     outputs=gr.Image(type="pil"),
#     title="Grayscale Image Converter",
#     description="Upload an image and see it converted to grayscale."
# )

# --- Gradio interface ---

with gr.Blocks(css="body {background: #f2f7ff;}") as image_transformation:
    gr.Markdown("## Image Operations\nUpload an image, apply effects (brightness, contrast, rotation, grayscale), or reset to the original image.")

    
    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Image")

    
    #with gr.Accordion("Adjust Brightness", open=False):
        #brightness_slider = gr.Slider(0.5, 1.5, value=1.0, step=0.1, label="Brightness")
        #brightness_output = gr.Image(type="pil", label="Brightness Output")
        #brightness_button = gr.Button("Apply Brightness")
        #brightness_button.click(adjust_brightness, inputs=[image_input, brightness_slider], outputs=brightness_output)

    #with gr.Accordion("Adjust Contrast", open=False):
        #contrast_slider = gr.Slider(0.5, 1.5, value=1.0, step=0.1, label="Contrast")
        #contrast_output = gr.Image(type="pil", label="Contrast Output")
        #contrast_button = gr.Button("Apply Contrast")
        #contrast_button.click(adjust_contrast, inputs=[image_input, contrast_slider], outputs=contrast_output)

    #with gr.Accordion("Rotate Image", open=False):
        #rotate_slider = gr.Slider(-180, 180, value=0, step=1, label="Rotation (degrees)")
        #rotate_output = gr.Image(type="pil", label="Rotation Output")
        #rotate_button = gr.Button("Apply Rotation")
        #rotate_button.click(rotate_image, inputs=[image_input, rotate_slider], outputs=rotate_output)

    #with gr.Accordion("Convert to Grayscale", open=False):
        #grayscale_output = gr.Image(type="pil", label="Grayscale Output")
        #grayscale_button = gr.Button("Apply Grayscale")
        #grayscale_button.click(to_grayscale, inputs=image_input, outputs=grayscale_output)

    with gr.Row():
        combined_brightness = gr.Slider(0.5, 1.5, value=1.0, step=0.1, label="Brightness")
        combined_contrast = gr.Slider(0.5, 1.5, value=1.0, step=0.1, label="Contrast")
        combined_rotation = gr.Slider(-180, 180, value=0, step=1, label="Rotation (degrees)")
        combined_grayscale = gr.Checkbox(False, label="Apply Grayscale")

    combined_output = gr.Image(type="pil", label="Final Combined Output")
    apply_combined = gr.Button("Apply Combined Effects")

    apply_combined.click(
        combined_effects,
        inputs=[image_input, combined_brightness, combined_contrast, combined_rotation, combined_grayscale],
        outputs=combined_output
    )

    # --- Bouton Reset ---
    gr.Markdown("### 🔄 Reset to Original Image")
    reset_button = gr.Button("Reset to Original")
    reset_button.click(
        reset_image,
        inputs=image_input,
        outputs=[combined_output],
    )

    #Save button 
    gr.Markdown("### Download Image")
    save_button = gr.Button("Save Image")
    file_output = gr.File(label="Download your edited image")
    save_button.click(
        save_image,
        inputs=combined_output,
        outputs=file_output
    )
