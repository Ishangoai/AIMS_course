import os
import re
from io import BytesIO

import numpy as np
from PIL import Image, ImageFont
from scipy import ndimage


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


def improved_remove_background(img, threshold=240):
    """Improved background removal using scipy (since rembg is not available)."""
    img = img.convert("RGBA")
    gray = img.convert("L")
    edges = ndimage.sobel(np.array(gray))
    mask = (np.array(gray) > threshold) | (edges > 50)
    data = np.array(img)
    data[mask, 3] = 0
    return Image.fromarray(data)


def image_to_ascii(img, width=100):
    """Convert image to ASCII art."""
    img = img.resize((width, int(width * img.height / img.width / 2)))
    img = img.convert("L")
    pixels = list(img.getdata())
    ascii_chars = ["@", "#", "S", "%", "?", "*", "+", ";", ":", ",", "."]
    ascii_img = "".join([ascii_chars[pixel // 25] for pixel in pixels])
    return "\n".join([ascii_img[i:i + width] for i in range(0, len(ascii_img), width)])


def generate_gif(img, frames=5):
    """Generate animated GIF from image with rotation."""
    gif_frames = []
    for i in range(frames):
        frame = img.rotate(i * (360 / frames), resample=Image.Resampling.BICUBIC, expand=False)
        gif_frames.append(frame)
    buffered = BytesIO()
    gif_frames[0].save(buffered, format="GIF", save_all=True, append_images=gif_frames[1:], duration=100, loop=0)
    return Image.open(buffered)


def _apply_crop(img, crop):
    """Apply crop transformation to image."""
    if crop < 1.0:
        w, h = img.size
        crop_w = int(w * crop)
        crop_h = int(h * crop)
        left = (w - crop_w) // 2
        top = (h - crop_h) // 2
        img = img.crop((left, top, left + crop_w, top + crop_h))
    return img


def _apply_pixelate(img, pixelate):
    """Apply pixelation effect to image."""
    if pixelate > 1:
        w, h = img.size
        small = img.resize((w // pixelate, h // pixelate), Image.Resampling.NEAREST)
        img = small.resize((w, h), Image.Resampling.NEAREST)
    return img


def get_font_paths():
    """Get dictionary of available font paths."""
    return {
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


def load_font(font_type, font_size):
    """Load font with fallback mechanism."""
    font_paths = get_font_paths()
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
        return font
    except (OSError, IOError) as e:
        print(f"Failed to load TrueType font {font_path}: {str(e)}")
        # Try other fonts as fallback
        for alt_font_type, alt_font_path in font_paths.items():
            if alt_font_path != font_path and os.path.exists(alt_font_path):
                try:
                    font = ImageFont.truetype(alt_font_path, int(font_size))
                    print(f"Fell back to alternative font: {alt_font_path} with size {font_size}")
                    return font
                except (OSError, IOError):
                    continue
        print("All TrueType fonts failed; using default PIL font (fixed size, ~10px)")
        return ImageFont.load_default()
