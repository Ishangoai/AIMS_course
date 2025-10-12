import random

import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps


def apply_basic_edits(img, grayscale, brightness, contrast, flip, blur):
    """Apply basic image edits."""
    if grayscale:
        img = img.convert("L").convert("RGB")

    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)

    if flip == "Horizontal":
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    elif flip == "Vertical":
        img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(blur))

    return img


def apply_intermediate_edits(img, sharpness, saturation):
    """Apply intermediate image edits."""
    img = ImageEnhance.Sharpness(img).enhance(sharpness)
    img = ImageEnhance.Color(img).enhance(saturation)
    return img


def _apply_sepia(img):
    """Apply sepia tone effect."""
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
    return sepia_img


def _apply_cartoon(img):
    """Apply cartoon effect."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    poster = ImageOps.posterize(img, 4)
    gray = poster.convert("L")
    edges = gray.filter(ImageFilter.SMOOTH_MORE).filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.invert(edges)
    edges_rgb = edges.convert("RGB")
    return ImageChops.multiply(poster, edges_rgb)


def _apply_glitch(img):
    """Apply glitch effect."""
    img_arr = np.array(img)
    for _ in range(10):
        row = random.randint(0, img_arr.shape[0] - 1)
        shift = random.randint(-20, 20)
        img_arr[row] = np.roll(img_arr[row], shift, axis=0)
    for c in range(3):
        shift = random.randint(-5, 5)
        img_arr[:, :, c] = np.roll(img_arr[:, :, c], shift, axis=0)
    return Image.fromarray(img_arr)


def apply_advanced_edits(img, sepia, edge_detect, cartoon, glitch, invert, emboss):
    """Apply advanced image edits."""
    if sepia:
        img = _apply_sepia(img)

    if edge_detect:
        img = img.convert("L").filter(ImageFilter.FIND_EDGES).convert("RGB")

    if cartoon:
        img = _apply_cartoon(img)

    if glitch:
        img = _apply_glitch(img)

    if invert:
        img = ImageOps.invert(img)

    if emboss:
        img = img.filter(ImageFilter.EMBOSS)

    return img


def apply_misc(img, rotation, opacity, text_overlay, text_color, text_font_size,
               text_position, font_type, make_gif, gif_frames):
    """Apply miscellaneous transformations."""
    try:
        from .image_utils import generate_gif
        from .text import add_text_overlay
    except ImportError:
        from image_utils import generate_gif
        from text import add_text_overlay

    img = img.rotate(rotation, resample=Image.Resampling.BICUBIC, expand=True)

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
