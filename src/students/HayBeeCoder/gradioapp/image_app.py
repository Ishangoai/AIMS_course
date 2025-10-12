import os
import random
import re
import tempfile
from io import BytesIO

import gradio as gr
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
from scipy import ndimage


# Improved background removal using scipy (since rembg is not available)
def improved_remove_background(img, threshold=240):
    img = img.convert("RGBA")
    gray = img.convert("L")
    edges = ndimage.sobel(np.array(gray))
    mask = (np.array(gray) > threshold) | (edges > 50)
    data = np.array(img)
    data[mask, 3] = 0
    return Image.fromarray(data)


def image_to_ascii(img, width=100):
    img = img.resize((width, int(width * img.height / img.width / 2)))
    img = img.convert("L")
    pixels = list(img.getdata())
    ascii_chars = ["@", "#", "S", "%", "?", "*", "+", ";", ":", ",", "."]
    ascii_img = "".join([ascii_chars[pixel // 25] for pixel in pixels])
    return "\n".join([ascii_img[i:i + width] for i in range(0, len(ascii_img), width)])


def generate_gif(img, frames=5):
    gif_frames = []
    for i in range(frames):
        frame = img.rotate(i * (360 / frames), resample=Image.BICUBIC, expand=False)
        gif_frames.append(frame)
    buffered = BytesIO()
    gif_frames[0].save(buffered, format="GIF", save_all=True, append_images=gif_frames[1:], duration=100, loop=0)
    return Image.open(buffered)


def parse_color(color):
    """Convert Gradio ColorPicker rgba string to RGB tuple of integers."""
    if isinstance(color, str) and color.startswith("rgba("):
        match = re.match(r"rgba\((\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\)", color)
        if match:
            r, g, b = int(float(match.group(1))), int(float(match.group(2))), int(float(match.group(3)))
            return (r, g, b)
    elif isinstance(color, str) and color.startswith("#"):
        color = color.lstrip("#")
        r, g, b = int(color[:2], 16), int(color[2:4], 16), int(color[4:6], 16)
        return (r, g, b)
    elif isinstance(color, tuple):
        return color[:3]
    return (0, 0, 0)


def add_text_overlay(img, text, color=(0, 0, 0), font_size=50, position="center", font_type="dejavusans"):  # noqa: C901
    if not text:
        return img
    draw = ImageDraw.Draw(img)

    # Font selection
    font = None
    font_paths = {
        "bebasnas": os.path.join(os.path.dirname(__file__), "fonts", "bebasnas.ttf"),
        "dejavusans": os.path.join(os.path.dirname(__file__), "fonts", "dejavusans.ttf"),
        "montserrat": os.path.join(os.path.dirname(__file__), "fonts", "montserrat.ttf"),
        "oswald": os.path.join(os.path.dirname(__file__), "fonts", "oswald.ttf"),
        "poppins": os.path.join(os.path.dirname(__file__), "fonts", "poppins.ttf"),
        "raleway": os.path.join(os.path.dirname(__file__), "fonts", "raleway.ttf"),
        "robotocondensed": os.path.join(os.path.dirname(__file__), "fonts", "robotocondensed.ttf"),
        "sourcesans": os.path.join(os.path.dirname(__file__), "fonts", "sourcesans.ttf"),
        "ubuntu": os.path.join(os.path.dirname(__file__), "fonts", "ubuntu.ttf")
    }
    fallback_font = os.path.join(os.path.dirname(__file__), "fonts", "dejavusans.ttf")

    font_path = font_paths.get(font_type, fallback_font)

    try:
        if not os.path.exists(font_path):
            print(f"Font file not found: {font_path}")
            font_path = fallback_font
            if not os.path.exists(font_path):
                raise OSError(f"Fallback font file not found: {font_path}")

        font = ImageFont.truetype(font_path, int(font_size))
        print(f"Loaded font: {font_path} with size {font_size}")
    except (OSError, IOError) as e:
        print(f"Failed to load TrueType font {font_path}: {str(e)}")
        # Try other fonts as fallback
        for alt_font_type, alt_font_path in font_paths.items():
            if alt_font_path != font_path and os.path.exists(alt_font_path):
                try:
                    font = ImageFont.truetype(alt_font_path, int(font_size))
                    print(f"Fell back to alternative font: {alt_font_path} with size {font_size}")
                    break
                except (OSError, IOError):
                    continue
        if font is None:
            print("All TrueType fonts failed; using default PIL font (fixed size, ~10px)")
            font = ImageFont.load_default()
            font_size = 10

    # Parse color
    color = parse_color(color)

    # Calculate text size and position
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # Determine position
    if position == "center":
        pos = ((img.width - w) // 2, (img.height - h) // 2)
    elif position == "top_left":
        pos = (0, 0)
    elif position == "top_right":
        pos = (img.width - w, 0)
    elif position == "bottom_left":
        pos = (0, img.height - h)
    elif position == "bottom_right":
        pos = (img.width - w, img.height - h)
    else:
        pos = ((img.width - w) // 2, (img.height - h) // 2)

    draw.text(pos, text, fill=color, font=font)
    return img


def kaleidoscope(img, segments=4):
    img = img.convert("RGB")
    w, h = img.size
    img_arr = np.array(img)
    result = np.zeros_like(img_arr)
    angle_step = 360 / segments
    center_x, center_y = w // 2, h // 2
    for x in range(w):
        for y in range(h):
            dx, dy = x - center_x, y - center_y
            angle = np.arctan2(dy, dx) * 180 / np.pi
            angle = angle % angle_step
            rad = np.sqrt(dx**2 + dy**2)
            new_x = int(center_x + rad * np.cos(angle * np.pi / 180))
            new_y = int(center_y + rad * np.sin(angle * np.pi / 180))
            if 0 <= new_x < w and 0 <= new_y < h:
                result[y, x] = img_arr[new_y, new_x]
    return Image.fromarray(result)


def wave_distortion(img, amplitude=10, wavelength=50):
    img = img.convert("RGB")
    img_arr = np.array(img)
    rows, cols, _ = img_arr.shape
    x = np.arange(cols)
    y = np.arange(rows)
    x, y = np.meshgrid(x, y)
    x_new = x + amplitude * np.sin(2 * np.pi * y / wavelength)
    y_new = y + amplitude * np.cos(2 * np.pi * x / wavelength)
    result = np.zeros_like(img_arr)
    for c in range(3):
        result[:, :, c] = ndimage.map_coordinates(img_arr[:, :, c], [y_new, x_new], order=1)
    return Image.fromarray(result.astype(np.uint8))


def channel_swap(img, swap_type="RB"):
    img = img.convert("RGB")
    r, g, b = img.split()
    if swap_type == "RB":
        img = Image.merge("RGB", (b, g, r))
    elif swap_type == "RG":
        img = Image.merge("RGB", (g, r, b))
    elif swap_type == "GB":
        img = Image.merge("RGB", (r, b, g))
    return img


def mosaic(img, tile_size=50):
    img = img.convert("RGB")
    w, h = img.size
    img_arr = np.array(img)
    result = img_arr.copy()
    tiles_x = w // tile_size
    tiles_y = h // tile_size
    for i in range(tiles_y):
        for j in range(tiles_x):
            if random.random() > 0.5:
                x = random.randint(0, tiles_x - 1) * tile_size
                y = random.randint(0, tiles_y - 1) * tile_size
                result[i * tile_size:(i + 1) * tile_size, j *
                       tile_size:(j + 1) * tile_size] = img_arr[y:y + tile_size, x:x + tile_size]
    return Image.fromarray(result)


def add_noise(img, noise_level=0.1):
    img = img.convert("RGB")
    img_arr = np.array(img)
    noise = np.random.normal(0, noise_level * 255, img_arr.shape)
    noisy_img = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)


def vignette(img, intensity=0.5):
    img = img.convert("RGB")
    w, h = img.size
    img_arr = np.array(img)
    x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    mask = 1 - intensity * (x**2 + y**2)
    mask = np.clip(mask, 0, 1)[:, :, np.newaxis]
    result = (img_arr * mask).astype(np.uint8)
    return Image.fromarray(result)


# --- Helper Functions --- #

def _apply_crop(img, crop):
    if crop < 1.0:
        w, h = img.size
        crop_w = int(w * crop)
        crop_h = int(h * crop)
        left = (w - crop_w) // 2
        top = (h - crop_h) // 2
        img = img.crop((left, top, left + crop_w, top + crop_h))
    return img


def _apply_pixelate(img, pixelate):
    if pixelate > 1:
        w, h = img.size
        small = img.resize((w // pixelate, h // pixelate), Image.NEAREST)
        img = small.resize((w, h), Image.NEAREST)
    return img


def _apply_basic_edits(img, grayscale, brightness, contrast, flip, blur):
    if grayscale:
        img = img.convert("L").convert("RGB")

    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)

    if flip == "Horizontal":
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif flip == "Vertical":
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(blur))

    return img


def _apply_intermediate_edits(img, sharpness, saturation):
    img = ImageEnhance.Sharpness(img).enhance(sharpness)
    img = ImageEnhance.Color(img).enhance(saturation)
    return img


def _apply_advanced_edits(img, sepia, edge_detect, cartoon, glitch, invert, emboss):  # noqa: C901
    if sepia:
        if img.mode != "RGB":
            img = img.convert("RGB")
        sepia_img = Image.new("RGB", img.size)
        for x in range(img.width):
            for y in range(img.height):
                r, g, b = img.getpixel((x, y))
                new_r = min(int(r * 0.393 + g * 0.769 + b * 0.189), 255)
                new_g = min(int(r * 0.349 + g * 0.686 + b * 0.168), 255)
                new_b = min(int(r * 0.272 + g * 0.534 + b * 0.131), 255)
                sepia_img.putpixel((x, y), (new_r, new_g, new_b))
        img = sepia_img

    if edge_detect:
        img = img.convert("L").filter(ImageFilter.FIND_EDGES).convert("RGB")

    if cartoon:
        if img.mode != "RGB":
            img = img.convert("RGB")
        poster = ImageOps.posterize(img, 4)
        gray = poster.convert("L")
        edges = gray.filter(ImageFilter.SMOOTH_MORE).filter(ImageFilter.FIND_EDGES)
        edges = ImageOps.invert(edges)
        edges_rgb = edges.convert("RGB")
        img = ImageChops.multiply(poster, edges_rgb)

    if glitch:
        img_arr = np.array(img)
        for _ in range(10):
            row = random.randint(0, img_arr.shape[0] - 1)
            shift = random.randint(-20, 20)
            img_arr[row] = np.roll(img_arr[row], shift, axis=0)
        for c in range(3):
            shift = random.randint(-5, 5)
            img_arr[:, :, c] = np.roll(img_arr[:, :, c], shift, axis=0)
        img = Image.fromarray(img_arr)

    if invert:
        img = ImageOps.invert(img)

    if emboss:
        img = img.filter(ImageFilter.EMBOSS)

    return img


def _apply_effects(img, apply_kaleidoscope, segments, apply_wave, wave_amplitude, wave_length,
                   channel_swap_type, apply_mosaic, tile_size, apply_noise, noise_level,
                   apply_vignette, vignette_intensity):
    if apply_kaleidoscope:
        img = kaleidoscope(img, segments)
    if apply_wave:
        img = wave_distortion(img, wave_amplitude, wave_length)
    if channel_swap_type != "None":
        img = channel_swap(img, channel_swap_type)
    if apply_mosaic:
        img = mosaic(img, tile_size)
    if apply_noise:
        img = add_noise(img, noise_level)
    if apply_vignette:
        img = vignette(img, vignette_intensity)
    return img


def _apply_misc(img, rotation, opacity, text_overlay, text_color, text_font_size,
                text_position, font_type, make_gif, gif_frames):
    img = img.rotate(rotation, resample=Image.BICUBIC, expand=True)

    if opacity < 1.0:
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.blend(bg, img, opacity)

    img = add_text_overlay(img, text_overlay, color=text_color, font_size=text_font_size,
                           position=text_position, font_type=font_type)

    if make_gif and gif_frames > 1:
        img = generate_gif(img, gif_frames)

    return img


# --- Main Function --- #

def edit_image(
    original, grayscale, brightness, contrast, rotation, flip, blur, sharpness, saturation, crop, pixelate,
    sepia, edge_detect, cartoon, glitch, invert, emboss, opacity, remove_bg, make_gif, gif_frames,
    text_overlay, text_color, text_font_size, text_position, font_type,
    apply_kaleidoscope, segments, apply_wave, wave_amplitude, wave_length,
    channel_swap_type, apply_mosaic, tile_size, apply_noise, noise_level,
    apply_vignette, vignette_intensity
):
    """Applies multiple image transformations and effects in a structured, modular way."""
    if original is None:
        return None, ""

    img = original.copy()

    # Preprocessing
    img = _apply_crop(img, crop)
    img = _apply_pixelate(img, pixelate)
    if remove_bg:
        img = improved_remove_background(img)

    # Main edit stages
    img = _apply_basic_edits(img, grayscale, brightness, contrast, flip, blur)
    img = _apply_intermediate_edits(img, sharpness, saturation)
    img = _apply_advanced_edits(img, sepia, edge_detect, cartoon, glitch, invert, emboss)
    img = _apply_effects(
        img, apply_kaleidoscope, segments, apply_wave, wave_amplitude, wave_length,
        channel_swap_type, apply_mosaic, tile_size, apply_noise, noise_level,
        apply_vignette, vignette_intensity
    )
    img = _apply_misc(
        img, rotation, opacity, text_overlay, text_color, text_font_size,
        text_position, font_type, make_gif, gif_frames
    )

    # Placeholder for ASCII output
    ascii_text = ""

    return img, ascii_text


def generate_ascii(original, *params):
    img = edit_image(original, *params)[0]
    if img is None:
        return ""
    return image_to_ascii(img)


def upload_image(image):
    return image, image, "", ""


def reset():
    return (False, 1.0, 1.0, 0, "None", 0, 1.0, 1.0, 1.0, 1, False, False, False, False, False, False, 1.0, False,
            False, 1, "", "#000000", 50, "center", "dejavusans", False, 4, False, 10, 50, "None", False, 50, False,
            0.1, False, 0.5)


def randomize():
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
    img = edit_image(original, *params)[0]
    if img is None:
        return None
    tmp_file = tempfile.NamedTemporaryFile(suffix=".png" if not params[-17] else ".gif", delete=False)
    img.save(tmp_file.name)
    return tmp_file.name


with gr.Blocks(title="Creative Image Editor", theme=gr.themes.Soft(), css="""
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

    def update_output(original, *params):
        img, _ = edit_image(original, *params)
        return img, ""

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

if __name__ == "__main__":
    app.launch()
