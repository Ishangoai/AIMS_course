from PIL import ImageDraw

try:
    from .image_utils import load_font, parse_color
except ImportError:
    from image_utils import load_font, parse_color


def add_text_overlay(img, text, color=(0, 0, 0), font_size=50, position="center",
                      font_type="dejavusans"):
    """Add text overlay to image."""
    if not text:
        return img
    draw = ImageDraw.Draw(img)

    # Load font
    font = load_font(font_type, font_size)

    # Parse color
    color = parse_color(color)

    # Calculate text size and position
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # Determine position
    if position == "center":
        pos = ((img.width - w) // 2, (img.height - h) // 2)
    elif position == "top_left":
        pos = (20, 0)
    elif position == "top_right":
        pos = (img.width - w - 20, 0)
    elif position == "bottom_left":
        pos = (0 + 10, img.height - h - 15)
    elif position == "bottom_right":
        pos = (img.width - w - 15, img.height - h - 15)
    else:
        pos = ((img.width - w) // 2, (img.height - h) // 2)

    draw.text(pos, text, fill=color, font=font)
    return img
