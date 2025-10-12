import tempfile
import random

import gradio as gr

try:
    from .processing import edit_image, generate_ascii
except ImportError:
    from processing import edit_image, generate_ascii


def upload_image(image):
    """Handle image upload."""
    return image, image, "", ""


def reset():
    """Reset all parameters to default values."""
    return (False, 1.0, 1.0, 0, "None", 0, 1.0, 1.0, 1.0, 1, False, False, False, False, False, False, 1.0, False,
            False, 1, "", "#000000", 50, "center", "dejavusans", False, 4, False, 10, 50, "None", False, 50, False,
            0.1, False, 0.5)


def randomize():
    """Randomize all parameters."""
    return (
        random.choice([True, False]),
        random.uniform(0.5, 1.5),
        random.uniform(0.5, 1.5),
        random.uniform(-90, 90),
        random.choice(["None", "Horizontal", "Vertical"]),
        random.uniform(0, 5),
        random.uniform(0.5, 1.5),
        random.uniform(0.5, 1.5),
        random.uniform(0.5, 1.0),
        random.randint(1, 10),
        random.choice([True, False]),
        random.choice([True, False]),
        random.choice([True, False]),
        random.choice([True, False]),
        random.choice([True, False]),
        random.choice([True, False]),
        random.uniform(0.7, 1.0),
        random.choice([True, False]),
        random.choice([True, False]),
        random.randint(1, 5),
        "",
        f"#{random.randint(0, 255):02x}{random.randint(0, 255):02x}{random.randint(0, 255):02x}",
        random.randint(20, 100),
        random.choice(["center", "top_left", "top_right", "bottom_left", "bottom_right"]),
        random.choice(["bebasnas", "dejavusans",
                       "montserrat", "oswald", "poppins", "raleway", "robotocondensed", "sourcesans",
                        "ubuntu"]),
        random.choice([True, False]),
        random.randint(2, 12),
        random.choice([True, False]),
        random.uniform(0, 20),
        random.uniform(10, 100),
        random.choice(["None", "RB", "RG", "GB"]),
        random.choice([True, False]),
        random.randint(10, 100),
        random.choice([True, False]),
        random.uniform(0, 0.5),
        random.choice([True, False]),
        random.uniform(0, 1)
    )


def apply_preset(preset):
    """Apply preset configurations."""
    (grayscale, brightness, contrast, rotation, flip, blur,
      sharpness, saturation, crop, pixelate, sepia, edge_detect,
     cartoon, glitch, invert, emboss, opacity, remove_bg, make_gif, gif_frames, text_overlay, text_color,
      text_font_size, text_position, font_type, apply_kaleidoscope, segments, apply_wave,
      wave_amplitude, wave_length, channel_swap_type, apply_mosaic, tile_size, apply_noise, noise_level,
        apply_vignette, vignette_intensity) = reset()
    if preset == "Vintage":
        brightness = 0.8
        contrast = 1.2
        saturation = 0.7
        sepia = True
    elif preset == "Dramatic":
        grayscale = True
        contrast = 1.5
        sharpness = 1.5
    elif preset == "Cartoon":
        cartoon = True
        saturation = 1.2
        sharpness = 1.2
    elif preset == "Glitchy":
        glitch = True
        brightness = 1.1
        contrast = 1.1
    elif preset == "Embossed Art":
        emboss = True
        grayscale = True
        sharpness = 1.5
    elif preset == "Kaleidoscope Dream":
        apply_kaleidoscope = True
        segments = 6
        saturation = 1.2
    elif preset == "Wavy Surreal":
        apply_wave = True
        wave_amplitude = 15
        wave_length = 60
        channel_swap_type = "RB"
    elif preset == "Mosaic Madness":
        apply_mosaic = True
        tile_size = 40
        apply_noise = True
        noise_level = 0.2
    return (grayscale, brightness, contrast, rotation, flip, blur,
            sharpness, saturation, crop, pixelate, sepia, edge_detect, cartoon, glitch, invert,
            emboss, opacity, remove_bg,
            make_gif, gif_frames, text_overlay, text_color, text_font_size, text_position, font_type,
            apply_kaleidoscope, segments, apply_wave, wave_amplitude, wave_length, channel_swap_type,
            apply_mosaic, tile_size, apply_noise,
            noise_level, apply_vignette, vignette_intensity)


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
    return img, ""


def create_interface():
    """Create the Gradio interface."""
    with gr.Blocks(title="Creative Image Editor", css="""
        .gradio-container .accordion .accordion-content {
            transition: max-height 0.3s ease-in-out, opacity 0.3s ease-in-out;
            max-height: 0;
            opacity: 0;
            overflow: hidden;
        }
        .gradio-container .accordion.open .accordion-content {
            max-height: 1000px; /* Adjust based on content size */
            opacity: 1;
        }
        .gradio-container .accordion .label-wrap {
            transition: background-color 0.2s ease;
        }
        .gradio-container .accordion .label-wrap:hover {
            background-color: rgba(0, 0, 0, 0.05);
        }
    """) as app:
        gr.Markdown("# Creative Image Editor")
        gr.Markdown("Unleash your creativity with unique filters, effects, and more!"
        " Now with kaleidoscope, wave distortion,"
        " channel swap, mosaic, noise, vignette effects, and customizable text overlay.")

        # Custom JavaScript to enforce single-open accordion behavior
        gr.HTML("""
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                const accordions = document.querySelectorAll('.accordion');
                accordions.forEach(accordion => {
                    const header = accordion.querySelector('.label-wrap');
                    header.addEventListener('click', function() {
                        const isOpen = accordion.classList.contains('open');
                        // Close all accordions
                        accordions.forEach(acc => {
                            if (acc !== accordion) {
                                acc.classList.remove('open');
                                const content = acc.querySelector('.accordion-content');
                                content.style.maxHeight = '0';
                                content.style.opacity = '0';
                            }
                        });
                        // Toggle the clicked accordion
                        if (!isOpen) {
                            accordion.classList.add('open');
                            const content = accordion.querySelector('.accordion-content');
                            content.style.maxHeight = content.scrollHeight + 'px';
                            content.style.opacity = '1';
                        } else {
                            accordion.classList.remove('open');
                            const content = accordion.querySelector('.accordion-content');
                            content.style.maxHeight = '0';
                            content.style.opacity = '0';
                        }
                    });
                });
            });
        </script>
        """)

        original = gr.State()

        input_img = gr.Image(type="pil", label="Upload Image")

        presets = gr.Dropdown(["None", "Vintage", "Dramatic", "Cartoon", "Glitchy", "Embossed Art", "Kaleidoscope Dream",
                               "Wavy Surreal", "Mosaic Madness"], label="Apply Preset", value="None",
                               info="Select a preset to quickly apply a unique style.")

        with gr.Accordion("Basic Features", open=False):
            with gr.Row():
                grayscale = gr.Checkbox(label="Grayscale", value=False, info="Convert image to black and white.")
                flip = gr.Radio(["None", "Horizontal", "Vertical"], label="Flip", value="None",
                                info="Flip the image horizontally"
                " or vertically.")

            with gr.Row():
                brightness = gr.Slider(0.5, 1.5, value=1.0, label="Brightness", info="Adjust the brightness level.")
                contrast = gr.Slider(0.5, 1.5, value=1.0, label="Contrast", info="Adjust the contrast level.")

            with gr.Row():
                rotation = gr.Slider(-180, 180, value=0, step=1, label="Rotation (degrees)",
                                     info="Rotate the image by degrees.")
                blur = gr.Slider(0, 10, value=0, step=0.5, label="Blur", info="Apply Gaussian blur.")

        with gr.Accordion("Intermediate Features", open=False):
            with gr.Row():
                sharpness = gr.Slider(0, 2, value=1.0, label="Sharpness", info="Enhance or soften edges.")
                saturation = gr.Slider(0, 2, value=1.0, label="Saturation", info="Adjust color intensity.")

            with gr.Row():
                crop = gr.Slider(0.1, 1.0, value=1.0, label="Crop (Percentage)", info="Crop from the center.")
                pixelate = gr.Slider(1, 20, value=1, step=1, label="Pixelate", info="Create a pixel art effect.")

        with gr.Accordion("Advanced Features", open=False):
            with gr.Row():
                sepia = gr.Checkbox(label="Sepia Tone", value=False, info="Apply a vintage sepia effect.")
                edge_detect = gr.Checkbox(label="Edge Detection", value=False, info="Highlight edges in the image.")
                cartoon = gr.Checkbox(label="Cartoonify", value=False, info="Turn image into a cartoon style.")

            with gr.Row():
                glitch = gr.Checkbox(label="Glitch Effect", value=False, info="Apply a digital glitch art effect.")
                invert = gr.Checkbox(label="Invert Colors", value=False, info="Invert all colors.")
                emboss = gr.Checkbox(label="Emboss", value=False, info="Apply an emboss filter for 3D effect.")

            with gr.Row():
                opacity = gr.Slider(0, 1, value=1.0, label="Opacity", info="Adjust transparency.")

        with gr.Accordion("Crazy Features", open=False):
            with gr.Row():
                apply_kaleidoscope = gr.Checkbox(label="Kaleidoscope", value=False,
                                                 info="Apply a symmetrical kaleidoscope effect.")
                segments = gr.Slider(2, 12, value=4, step=2, label="Kaleidoscope Segments",
                                      info="Number of mirrored segments.")
                apply_wave = gr.Checkbox(label="Wave Distortion", value=False, info="Apply a wavy distortion effect.")
            with gr.Row():
                wave_amplitude = gr.Slider(0, 20, value=10, label="Wave Amplitude", info="Strength of the wave effect.")
                wave_length = gr.Slider(10, 100, value=50, label="Wave Length", info="Length of the waves.")
                channel_swap_type = gr.Dropdown(["None", "RB", "RG", "GB"], label="Channel Swap", value="None",
                                                 info="Swap color channels for a surreal effect.")
            with gr.Row():
                apply_mosaic = gr.Checkbox(label="Mosaic Effect", value=False,
                                            info="Shuffle image tiles for a mosaic effect.")
                tile_size = gr.Slider(10, 100, value=50, step=10, label="Tile Size", info="Size of mosaic tiles.")
                apply_noise = gr.Checkbox(label="Add Noise", value=False, info="Add random grainy noise.")
            with gr.Row():
                noise_level = gr.Slider(0, 0.5, value=0.1, label="Noise Level", info="Intensity of the noise.")
                apply_vignette = gr.Checkbox(label="Vignette", value=False, info="Darken edges for a vignette effect.")
                vignette_intensity = gr.Slider(0, 1, value=0.5, label="Vignette Intensity",
                                                info="Strength of the vignette.")

        with gr.Accordion("Add and Edit Text", open=False):
            with gr.Row():
                remove_bg = gr.Checkbox(label="Remove Background", value=False,
                                         info="Remove white-ish background using edge detection.", visible=False)
                make_gif = gr.Checkbox(label="GIF-fy", value=False, info="Turn into animated GIF with rotation.",
                                       visible=False)
                gif_frames = gr.Slider(1, 10, value=1, step=1, label="GIF Frames",
                                       info="Number of frames for GIF (if GIF-fy enabled).",
                                       visible=False)

            with gr.Row():
                text_overlay = gr.Textbox(label="Text Overlay", value="", info="Add text to the image.")
                text_color = gr.ColorPicker(label="Text Color", value="#000000",
                                             info="Select the color of the text overlay.")
                text_font_size = gr.Slider(10, 200, value=50, step=5, label="Text Font Size", info="Adjust the font size of"
                " the overlay text.")
                text_position = gr.Dropdown(["center", "top_left", "top_right", "bottom_left", "bottom_right"],
                                             label="Text Position", value="center", info="Select the position of the text o"
                                             " the image.")
                font_type = gr.Dropdown(
                    ["bebasnas", "dejavusans", "montserrat", "oswald", "poppins", "raleway", "robotocondensed",
                     "sourcesans", "ubuntu"],
                    label="Font Type",
                    value="dejavusans",
                    info="Select the font for the text overlay (requires fonts in project folder)."
                )

        output_img = gr.Image(label="Edited Image", interactive=False)

        ascii_output = gr.Textbox(label="ASCII Art", interactive=False, lines=10)

        with gr.Row():
            reset_btn = gr.Button("Reset")
            randomize_btn = gr.Button("Randomize")
            ascii_btn = gr.Button("Generate ASCII Art")
            download_btn = gr.Button("Download Edited Image")
            cloud_btn = gr.Button("Upload to Cloud (Coming Soon)", interactive=False)

        download_file = gr.File(label="Download File", visible=True)
        cloud_url = gr.Textbox(label="Shareable Cloud URL", interactive=False, placeholder="Cloud upload coming soon.")

        # Event handlers
        input_img.upload(upload_image, input_img, [original, output_img, ascii_output, cloud_url])

        params = [
            grayscale, brightness, contrast, rotation, flip, blur, sharpness, saturation, crop, pixelate,
            sepia, edge_detect, cartoon, glitch, invert, emboss, opacity, remove_bg, make_gif, gif_frames,
            text_overlay, text_color, text_font_size, text_position, font_type, apply_kaleidoscope, segments,
            apply_wave, wave_amplitude, wave_length, channel_swap_type, apply_mosaic, tile_size, apply_noise,
            noise_level, apply_vignette, vignette_intensity
        ]
        all_inputs = [original] + params

        for comp in params:
            comp.change(
                update_output,
                all_inputs,
                [output_img, cloud_url]
            )

        reset_btn.click(
            reset,
            None,
            params
        ).then(
            update_output,
            all_inputs,
            [output_img, cloud_url]
        )

        randomize_btn.click(
            randomize,
            None,
            params
        ).then(
            update_output,
            all_inputs,
            [output_img, cloud_url]
        )

        presets.change(
            apply_preset,
            presets,
            params
        ).then(
            update_output,
            all_inputs,
            [output_img, cloud_url]
        )

        ascii_btn.click(
            generate_ascii,
            all_inputs,
            ascii_output
        )

        download_btn.click(
            download_image,
            all_inputs,
            download_file
        )

    return app
