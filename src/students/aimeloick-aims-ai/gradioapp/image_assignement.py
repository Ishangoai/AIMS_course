import gradio as gr
from gradioapp.utils.image_assignment_utils import transform_image

# ==============================
# CSS
# ==============================
css = """
.shadow-box {
    border: 2px solid #ddd;
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    padding: 10px;
    border-radius: 8px;
    background-color: white;
}
"""


# ==============================
# Functions
# ==============================
def transform_images(image, grayscale_or_not, brightness, contrast_factor, degree, threshold):
    """Apply transformations to an image."""
    if image is None:
        return None
    return transform_image(image, grayscale_or_not, brightness, contrast_factor, degree, threshold)


def reset_image():
    """Reset all image transformation controls to default."""
    return "No Grayscale", 1.0, 1.0, 0, 80


def download_image(image):
    """Return the image for download."""
    if image is None:
        gr.Warning("No image to download!")
        return None
    return image


def process_folder(folder):
    """Process the uploaded folder and return list of files."""
    if not folder:
        return "No folder uploaded"
    file_list = [d.name for d in folder]
    return f"Files uploaded: {', '.join(file_list)}"


# ==============================
# Gradio Interface
# ==============================
with gr.Blocks(css=css) as app:
    gr.HTML("<hr>")
    gr.Markdown(
        '<center><h2>🖼️ Image Editor — Grayscale, Brightness, Contrast & Rotation</h2></center>'
    )
    gr.HTML("<hr>")

    with gr.Row():
        gr.Markdown("# Please upload an image.")

    with gr.Row():
        # ---------- Left Column (Image Processing) ----------
        with gr.Column(elem_classes="shadow-box"):
            with gr.Row():
                with gr.Column():
                    image = gr.Image(label="Image", type="pil")
                with gr.Column():
                    output_img = gr.Image(label="Résultat", visible=True)

            grayscale_or_not = gr.Radio(
                choices=["Grayscale", "No Grayscale"],
                value="No Grayscale",
                label="Grayscale or Not",
            )
            brightness = gr.Slider(0.5, 1.5, value=1, label="Brightness")
            contrast_factor = gr.Slider(0.5, 1.5, value=1, label="Contrast")
            degree = gr.Slider(-180, 180, value=0, label="Rotation")
            threshold = gr.Slider(100, 300, value=80, label="Threshold")

            controls = [image, grayscale_or_not, brightness, contrast_factor, degree, threshold]

            # Update output when controls change
            for ctrl in controls:
                ctrl.change(transform_images, inputs=controls, outputs=output_img)

            with gr.Row():
                with gr.Column():
                    reset_btn = gr.Button("Reset")
                    reset_btn.click(
                        fn=reset_image,
                        outputs=[grayscale_or_not, brightness, contrast_factor, degree, threshold],
                    )
                with gr.Column():
                    # File component for download (hidden by default)
                    download_file = gr.File(label="Download Image", visible=False)
                    download_btn = gr.Button("Download")
                    download_btn.click(
                        fn=download_image,
                        inputs=[output_img],
                        outputs=[download_file]
                    )
