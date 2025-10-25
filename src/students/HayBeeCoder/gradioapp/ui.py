import tempfile

import gradio as gr

from .css import HEADER_CSS, TAB_CSS, UI_CSS

try:
    from .processing import edit_image, join_images
except ImportError:
    from processing import edit_image, join_images


def upload_image(file_path):
    """Handle image upload."""
    if file_path is None:
        return None, None, ""

    from PIL import Image

    try:
        # Open the image from the file path
        image = Image.open(file_path)
        return image, image, ""
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None, ""


def reset():
    """Reset all parameters to default values."""
    return (
        False, 1.0, 1.0, 0, "None", 0, 1.0, 1.0, 1.0, 1, False, False, False, False, False, False, 1.0,
        False, False, 1, "", "#000000", 50, "center", "dejavusans", False, 4, False, 10, 50, "None",
        False, 50, False, 0.1, False, 0.5,
    )


def download_image(original, *params):
    """Download edited image."""
    img = edit_image(original, *params)[0]
    if img is None:
        return None
    tmp_file = tempfile.NamedTemporaryFile(suffix=".png" if not params[-17] else ".gif", delete=False)
    img.save(tmp_file.name)
    return tmp_file.name


def update_output(original, *params):
    """Update output image."""
    img, _ = edit_image(original, *params)
    return img


def create_interface():
    """Create the Gradio interface."""
    with gr.Blocks(
        title="Imagaims",
        theme="dark",
        css=f"{UI_CSS}",


    ) as app:
        # Enhanced Header with Icons and Animations
        gr.HTML(f"{HEADER_CSS}")

        # Create Tabbed Interface
        with gr.Tabs(elem_classes=["main-tabs"]):
            with gr.Tab("🎨 Image Editor", elem_classes=["tab-content"]):
                gr.HTML(TAB_CSS)

                original = gr.State()

                # Main Layout: Images on Left, Features on Right
                with gr.Row(elem_classes=["main-layout"]):
                    # Left Side: Image Display Area
                    with gr.Column(scale=2, elem_classes=["image-display-area"]):
                        # Image Upload Section
                        input_img = gr.UploadButton(
                            "📁 Upload Image", file_types=["image"], file_count="single", elem_classes=["upload-button"]
                        )

                        # Live Preview
                        gr.HTML("""
                        <div style="text-align: center; margin: 2rem 0;">
                            <h3 style="color: #667eea; font-weight: 600; margin-bottom: 1.5rem;">Live Preview</h3>
                        </div>
                        """)

                        output_img = gr.Image(label="Edited Image", interactive=False, elem_classes=["image-container"])

                        # Download File Widget
                        download_file = gr.File(label="Download File", visible=True)

                    # Right Side: Scrollable Features Panel
                    with gr.Column(scale=1, elem_classes=["features-panel"]):
                        gr.HTML("""
                        <div style="text-align: center; margin: 1rem 0;">
                            <h3 style="color: #667eea; font-weight: 600; margin-bottom: 0.5rem;">🎨 Image Effects</h3>
                            <p style="color: #6c757d; font-size: 0.9rem; margin: 0;">
                            Adjust settings and see changes instantly</p>
                        </div>
                        """)

                        # Scrollable Features Container
                        with gr.Column(elem_classes=["scrollable-features"]):
                            with gr.Accordion("Basic Features", open=True):
                                gr.HTML("""
                                <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%,
                                 rgba(118, 75, 162, 0.1) 100%);
                                            padding: 1rem; border-radius: 12px; margin-bottom: 1.5rem;
                                            border-left: 4px solid #667eea;">
                                    <p style="margin: 0; color: #667eea; font-weight: 600;">
                                    Essential adjustments for perfect
                                    image enhancement</p>
                                </div>
                                """)
                                with gr.Row():
                                    grayscale = gr.Checkbox(
                                        label="Grayscale", value=False, info="Convert image to black and white."
                                    )
                                    flip = gr.Radio(
                                        ["None", "Horizontal", "Vertical"],
                                        label="Flip",
                                        value="None",
                                        info="Flip the image horizontally or vertically.",
                                    )

                                with gr.Row():
                                    brightness = gr.Slider(
                                        0.5, 1.5, value=1.0, label="Brightness", info="Adjust the brightness level."
                                    )
                                    contrast = gr.Slider(
                                        0.5, 1.5, value=1.0, label="Contrast", info="Adjust the contrast level."
                                    )

                                with gr.Row():
                                    rotation = gr.Slider(
                                        -180,
                                        180,
                                        value=0,
                                        step=1,
                                        label="Rotation (degrees)",
                                        info="Rotate the image by degrees.",
                                    )
                                    blur = gr.Slider(
                                        0, 10, value=0, step=0.5, label="Blur", info="Apply Gaussian blur."
                                    )

                            with gr.Accordion("Intermediate Features", open=False):
                                gr.HTML("""
                                <div style="background:
                                linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%);
                                            padding: 1rem; border-radius: 12px; margin-bottom: 1.5rem;
                                             border-left: 4px solid #10b981;">
                                    <p style="margin: 0; color: #10b981; font-weight: 600;">Advanced adjustments
                                    for professional results</p>
                                </div>
                                """)
                                with gr.Row():
                                    sharpness = gr.Slider(
                                        0, 2, value=1.0, label="Sharpness", info="Enhance or soften edges."
                                    )
                                    saturation = gr.Slider(
                                        0, 2, value=1.0, label="Saturation", info="Adjust color intensity."
                                    )

                                with gr.Row():
                                    crop = gr.Slider(
                                        0.1, 1.0, value=1.0, label="Crop (Percentage)", info="Crop from the center."
                                    )
                                    pixelate = gr.Slider(
                                        1, 20, value=1, step=1, label="Pixelate", info="Create a pixel art effect."
                                    )

                            with gr.Accordion("Advanced Features", open=False):
                                gr.HTML("""
                                <div style="background:
                                linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.1) 100%);
                                            padding: 1rem; border-radius: 12px; margin-bottom: 1.5rem;
                                            border-left: 4px solid #ef4444;">
                                    <p style="margin: 0; color: #ef4444; font-weight: 600;">Creative effects
                                    that transform your images into art</p>
                                </div>
                                """)
                                with gr.Row():
                                    sepia = gr.Checkbox(
                                        label="Sepia Tone", value=False, info="Apply a vintage sepia effect."
                                    )
                                    edge_detect = gr.Checkbox(
                                        label="Edge Detection", value=False, info="Highlight edges in the image."
                                    )
                                    cartoon = gr.Checkbox(
                                        label="Cartoonify", value=False, info="Turn image into a cartoon style."
                                    )

                                with gr.Row():
                                    glitch = gr.Checkbox(
                                        label="Glitch Effect", value=False, info="Apply a digital glitch art effect."
                                    )
                                    invert = gr.Checkbox(label="Invert Colors", value=False, info="Invert all colors.")
                                    emboss = gr.Checkbox(
                                        label="Emboss", value=False, info="Apply an emboss filter for 3D effect."
                                    )

                                with gr.Row():
                                    opacity = gr.Slider(0, 1, value=1.0, label="Opacity", info="Adjust transparency.")

                            with gr.Accordion("🤯 Features", open=False):
                                gr.HTML("""
                                <div style="background:
                                linear-gradient(135deg, rgba(168, 85, 247, 0.1) 0%, rgba(139, 69, 19, 0.1) 100%);
                                    padding: 1rem; border-radius: 12px; margin-bottom: 1.5rem;
                                    border-left: 4px solid #a855f7;">
                                    <p style="margin: 0; color: #a855f7; font-weight: 600;">
                                     Mind-bending effects that push creative boundaries</p>
                                </div>
                                """)
                                with gr.Row():
                                    apply_kaleidoscope = gr.Checkbox(
                                        label="Kaleidoscope",
                                        value=False,
                                        info="Apply a symmetrical kaleidoscope effect.",
                                    )
                                    segments = gr.Slider(
                                        2,
                                        12,
                                        value=4,
                                        step=2,
                                        label="Kaleidoscope Segments",
                                        info="Number of mirrored segments.",
                                    )
                                    apply_wave = gr.Checkbox(
                                        label="Wave Distortion", value=False, info="Apply a wavy distortion effect."
                                    )
                                with gr.Row():
                                    wave_amplitude = gr.Slider(
                                        0, 20, value=10, label="Wave Amplitude", info="Strength of the wave effect."
                                    )
                                    wave_length = gr.Slider(
                                        10, 100, value=50, label="Wave Length", info="Length of the waves."
                                    )
                                    channel_swap_type = gr.Dropdown(
                                        ["None", "RB", "RG", "GB"],
                                        label=" Channel Swap",
                                        value="None",
                                        info="Swap color channels for a surreal effect.",
                                    )
                                with gr.Row():
                                    apply_mosaic = gr.Checkbox(
                                        label="Mosaic Effect",
                                        value=False,
                                        info="Shuffle image tiles for a mosaic effect.",
                                    )
                                    tile_size = gr.Slider(
                                        10, 100, value=50, step=10, label="Tile Size", info="Size of mosaic tiles."
                                    )
                                    apply_noise = gr.Checkbox(
                                        label="Add Noise", value=False, info="Add random grainy noise."
                                    )
                                with gr.Row():
                                    noise_level = gr.Slider(
                                        0, 0.5, value=0.1, label="Noise Level", info="Intensity of the noise."
                                    )
                                    apply_vignette = gr.Checkbox(
                                        label=" Vignette", value=False, info="Darken edges for a vignette effect."
                                    )
                                    vignette_intensity = gr.Slider(
                                        0, 1, value=0.5, label=" Vignette Intensity", info="Strength of the vignette."
                                    )

                            with gr.Accordion("Add and Edit Text", open=False):
                                gr.HTML("""
                                <div style="background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%,
                                 rgba(217, 119, 6, 0.1) 100%);
                                            padding: 1rem; border-radius: 12px; margin-bottom: 1.5rem;
                                             border-left: 4px solid #f59e0b;">
                                    <p style="margin: 0; color: #f59e0b; font-weight: 600;">
                                    Add beautiful typography to your images</p>
                                </div>
                                """)
                                with gr.Row():
                                    remove_bg = gr.Checkbox(
                                        label="Remove Background",
                                        value=False,
                                        info="Remove white-ish background using edge detection.",
                                        visible=False,
                                    )
                                    make_gif = gr.Checkbox(
                                        label="GIF-fy",
                                        value=False,
                                        info="Turn into animated GIF with rotation.",
                                        visible=False,
                                    )
                                    gif_frames = gr.Slider(
                                        1,
                                        10,
                                        value=1,
                                        step=1,
                                        label="GIF Frames",
                                        info="Number of frames for GIF (if GIF-fy enabled).",
                                        visible=False,
                                    )

                                with gr.Row():
                                    text_overlay = gr.Textbox(
                                        label=" Text Overlay", value="", info="Add text to the image."
                                    )
                                    text_color = gr.ColorPicker(
                                        label=" Text Color",
                                        value="#000000",
                                        info="Select the color of the text overlay.",
                                    )
                                    text_font_size = gr.Slider(
                                        10,
                                        200,
                                        value=50,
                                        step=5,
                                        label=" Text Font Size",
                                        info="Adjust the font size of the overlay text.",
                                    )
                                    text_position = gr.Dropdown(
                                        ["center", "top_left", "top_right", "bottom_left", "bottom_right"],
                                        label="Text Position",
                                        value="center",
                                        info="Select the position of the text on the image.",
                                    )
                                font_type = gr.Dropdown(
                                    [
                                        "bebasnas",
                                        "dejavusans",
                                        "montserrat",
                                        "oswald",
                                        "poppins",
                                        "raleway",
                                        "robotocondensed",
                                        "sourcesans",
                                        "ubuntu",
                                    ],
                                    label="Font Type",
                                    value="dejavusans",
                                    info="Select the font for the text overlay (requires fonts in project folder).",
                                )

                            # Control Buttons
                            with gr.Row():
                                reset_btn = gr.Button("Reset All", elem_classes=["btn", "secondary"])
                                download_btn = gr.Button("💾 Download", elem_classes=["btn", "primary"])

                # Event handlers
                input_img.upload(upload_image, input_img, [original, output_img])

                params = [grayscale, brightness, contrast, rotation, flip, blur, sharpness, saturation, crop, pixelate,
                    sepia, edge_detect, cartoon, glitch, invert, emboss, opacity, remove_bg, make_gif, gif_frames,
                    text_overlay, text_color, text_font_size, text_position, font_type, apply_kaleidoscope, segments,
                    apply_wave, wave_amplitude, wave_length, channel_swap_type, apply_mosaic, tile_size, apply_noise,
                    noise_level, apply_vignette, vignette_intensity,
                ]
                all_inputs = [original] + params

                for comp in params:
                    comp.change(update_output, all_inputs, [output_img])

                reset_btn.click(reset, None, params).then(update_output, all_inputs, [output_img])

                download_btn.click(download_image, all_inputs, download_file)

            with gr.Tab("🔗 Image Joiner", elem_classes=["tab-content"]):
                # Image Joining Section
                gr.HTML("""
                <div class="image-joiner-section" style="text-align: center; margin: 3rem 0 2rem 0; padding: 2rem;
                 background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                  border-radius: 20px;
                  border: 2px solid rgba(102, 126, 234, 0.2);">
                    <h2 style="color: #667eea; font-weight: 700; margin-bottom: 1rem;
                     font-size: 2rem;">🖼️ Image Joiner</h2>
                    <p style="color: #64748b; font-size: 1.1rem; margin: 0; max-width: 600px; margin: 0 auto;">
                    Combine two images into one! Upload two images and choose whether to join them
                    horizontally
                    (side by side) or vertically (stacked).</p>
                </div>
                """)

                # Image Joining Interface
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML("""
                        <div class="image-joiner-section" style="text-align: center; margin: 1rem 0;">
                            <h3 style="color: #667eea; font-weight: 600; margin-bottom: 0.5rem;">First Image</h3>
                            <p style="color: #6c757d; font-size: 0.9rem; margin: 0;">Upload your first image</p>
                        </div>
                        """)
                        join_img1 = gr.Image(type="pil", label="", height=200, container=False)

                    with gr.Column(scale=1):
                        gr.HTML("""
                        <div class="image-joiner-section" style="text-align: center; margin: 1rem 0;">
                            <h3 style="color: #667eea; font-weight: 600; margin-bottom: 0.5rem;">Second Image</h3>
                            <p style="color: #6c757d; font-size: 0.9rem; margin: 0;">Upload your second image</p>
                        </div>
                        """)
                        join_img2 = gr.Image(type="pil", label="", height=200, container=False)

                # Join Options
                with gr.Row():
                    with gr.Column(scale=1):
                        join_direction = gr.Radio(
                            ["horizontal", "vertical"],
                            label="Join Direction",
                            value="horizontal",
                            info="Choose how to combine the images",
                        )
                    with gr.Column(scale=1):
                        join_btn = gr.Button("🔗 Join Images", elem_classes=["btn", "primary"], size="lg")

                # Join Result Display
                gr.HTML("""
                <div class="image-joiner-section" style="text-align: center; margin: 2rem 0;">
                    <h3 style="color: #667eea; font-weight: 600; margin-bottom: 1.5rem;">Joined Image Result</h3>
                </div>
                """)

                joined_result = gr.Image(label="Joined Image", interactive=False, elem_classes=["image-container"])

                # Download joined image
                with gr.Row():
                    download_joined_btn = gr.Button("💾 Download Joined Image", elem_classes=["btn", "primary"])

                download_joined_file = gr.File(label="Download Joined File", visible=True)

                # Join Images Function
                def process_join_images(img1, img2, direction):
                    """Process image joining."""
                    if img1 is None or img2 is None:
                        return None, "Please upload both images to join them."

                    try:
                        joined_img = join_images(img1, img2, direction)
                        if joined_img is None:
                            return None, "Error joining images. Please try again."
                        return joined_img, "Images successfully joined!"
                    except Exception as e:
                        return None, f"Error joining images: {str(e)}"

                # Download joined image function
                def download_joined_image(joined_img):
                    """Download the joined image."""
                    if joined_img is None:
                        return None

                    import numpy as np
                    from PIL import Image

                    # Convert numpy array to PIL Image if needed
                    if isinstance(joined_img, np.ndarray):
                        if joined_img.dtype != np.uint8:
                            joined_img = (joined_img * 255).astype(np.uint8)
                        joined_img = Image.fromarray(joined_img)

                    tmp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    joined_img.save(tmp_file.name)
                    return tmp_file.name

                # Event handlers for image joining
                join_btn.click(
                    process_join_images,
                    [join_img1, join_img2, join_direction],
                    [joined_result, gr.Textbox(visible=False)],  # Hidden textbox for status
                )

                download_joined_btn.click(download_joined_image, joined_result, download_joined_file)

    return app
