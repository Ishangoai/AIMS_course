import gradio as gr
from gradioapp.utils.image_assignment_utils import download_single_image, transform_folder, transform_single_image

# ==============================
# CSS
# ==============================
css = """
.shadow-box {
    border: 2px solid #ddd;
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    padding: 20px;
    border-radius: 8px;
    background-color: white;
}
"""

# ==============================
# Gradio Interface
# ==============================
with gr.Blocks(css=css) as app:
    gr.HTML("<hr>")
    gr.Markdown('<center><h1>Image Transformer</h1></center>')
    gr.HTML("<hr>")

    # ==============================
    # TAB 1: Single Image Processing
    # ==============================
    with gr.Tab("Single Image"):
        gr.Markdown("### Transform a single image")

        with gr.Row():
            with gr.Column(scale=1, elem_classes="shadow-box"):
                single_image = gr.Image(label="Input Image", type="pil")

                single_grayscale = gr.Radio(
                    choices=["Grayscale", "No Grayscale"],
                    value="No Grayscale",
                    label="Grayscale",
                )
                single_brightness = gr.Slider(0.5, 1.5, value=1.0, step=0.1, label="Brightness")
                single_contrast = gr.Slider(0.5, 1.5, value=1.0, step=0.1, label="Contrast")
                single_degree = gr.Slider(-180, 180, value=0, step=1, label="Rotation (degrees)")
                single_remove_bg = gr.Checkbox(label="Remove White Background", value=False)
                single_threshold = gr.Slider(100, 255, value=200, step=5, label="Threshold (for background removal)")

            with gr.Column(scale=1, elem_classes="shadow-box"):
                single_output = gr.Image(label="Result")

                with gr.Row():
                    single_reset_btn = gr.Button("Reset", scale=1)
                    single_download_btn = gr.Button("Download", scale=1)

                single_download_file = gr.File(label="Download image here")

        # Update output when image or controls change
        def update_single_output(img, gs, br, co, deg, th, rb):
            return transform_single_image(img, gs, br, co, deg, th, rb)

        single_image.change(
            fn=update_single_output,
            inputs=[single_image, single_grayscale, single_brightness, single_contrast,
                   single_degree, single_threshold, single_remove_bg],
            outputs=single_output
        )

        single_grayscale.change(
            fn=update_single_output,
            inputs=[single_image, single_grayscale, single_brightness, single_contrast,
                   single_degree, single_threshold, single_remove_bg],
            outputs=single_output
        )

        single_brightness.change(
            fn=update_single_output,
            inputs=[single_image, single_grayscale, single_brightness, single_contrast,
                   single_degree, single_threshold, single_remove_bg],
            outputs=single_output
        )

        single_contrast.change(
            fn=update_single_output,
            inputs=[single_image, single_grayscale, single_brightness, single_contrast,
                   single_degree, single_threshold, single_remove_bg],
            outputs=single_output
        )

        single_degree.change(
            fn=update_single_output,
            inputs=[single_image, single_grayscale, single_brightness, single_contrast,
                   single_degree, single_threshold, single_remove_bg],
            outputs=single_output
        )

        single_threshold.change(
            fn=update_single_output,
            inputs=[single_image, single_grayscale, single_brightness, single_contrast,
                   single_degree, single_threshold, single_remove_bg],
            outputs=single_output
        )

        single_remove_bg.change(
            fn=update_single_output,
            inputs=[single_image, single_grayscale, single_brightness, single_contrast,
                   single_degree, single_threshold, single_remove_bg],
            outputs=single_output
        )

        single_reset_btn.click(
            fn=lambda: ["No Grayscale", 1.0, 1.0, 0, False, 200],
            outputs=[single_grayscale, single_brightness, single_contrast,
                    single_degree, single_remove_bg, single_threshold]
        )

        single_download_btn.click(
            fn=download_single_image,
            inputs=[single_output],
            outputs=[single_download_file]
        )

    # ==============================
    # TAB 2: Batch Processing
    # ==============================
    with gr.Tab("Batch Processing"):
        gr.Markdown("### Transform multiple images at once")

        with gr.Row():
            with gr.Column(scale=1, elem_classes="shadow-box"):
                gr.Markdown("#### Upload & Settings")

                batch_files = gr.File(
                    file_count="multiple",
                    label="Select multiple images or drag a folder",
                    file_types=["image"],
                    type="filepath"
                )

                gr.Markdown("*Tip: You can select multiple files or drag and drop an entire folder*")

                batch_grayscale = gr.Radio(
                    ["Grayscale", "No Grayscale"],
                    value="No Grayscale",
                    label="Grayscale"
                )
                batch_brightness = gr.Slider(0.5, 1.5, value=1.0, step=0.1, label="Brightness")
                batch_contrast = gr.Slider(0.5, 1.5, value=1.0, step=0.1, label="Contrast")
                batch_degree = gr.Slider(-180, 180, value=0, step=1, label="Rotation (degrees)")
                batch_remove_bg = gr.Checkbox(label="Remove White Background", value=False)
                batch_threshold = gr.Slider(100, 255, value=200, step=5, label="Threshold (for background removal)")

                with gr.Row():
                    batch_reset_btn = gr.Button("Reset", scale=1)
                    batch_submit_btn = gr.Button("Transform All Images", scale=2, variant="primary")

            with gr.Column(scale=1, elem_classes="shadow-box"):
                gr.Markdown("#### Results Preview")

                # Aperçu d'une image transformée (la première du batch)
                batch_preview_image = gr.Image(
                    label="Preview of first transformed image",
                    type="pil",
                    visible=True
                )

                # Statut du batch complet
                batch_output_text = gr.Textbox(
                    label="Processing Status",
                    lines=10,
                    interactive=False,
                    placeholder="Upload images and click 'Transform All Images' to start..."
                )

                # Bouton de téléchargement ZIP
                batch_download_zip = gr.File(
                    label="Download ZIP file with all transformed images"
                )

                # Instructions d'utilisation
                gr.Markdown("""
                **How to use:**
                1. **Upload images**: Click to select multiple files OR drag and drop a folder
                2. **Configure settings**: Adjust sliders and options as needed
                3. **Click "Transform All Images"**: Process all uploaded images
                4. **Download ZIP**: Get all transformed images in one compressed file

                **Supported formats**: JPG, JPEG, PNG, BMP, GIF, TIFF, WEBP
                """)

        batch_reset_btn.click(
            fn=lambda: ["No Grayscale", 1.0, 1.0, 0, False, 200],
            outputs=[batch_grayscale, batch_brightness, batch_contrast,
                    batch_degree, batch_remove_bg, batch_threshold]
        )

        batch_submit_btn.click(
            fn=transform_folder,
            inputs=[
                batch_files, batch_grayscale, batch_brightness,
                batch_contrast, batch_degree, batch_threshold, batch_remove_bg
            ],
            outputs=[batch_output_text, batch_download_zip]
        )
