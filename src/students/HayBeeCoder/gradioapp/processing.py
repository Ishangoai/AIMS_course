try:
    from .effects import apply_effects
    from .filters import apply_advanced_edits, apply_basic_edits, apply_intermediate_edits, apply_misc
    from .image_utils import _apply_crop, _apply_pixelate, image_to_ascii, improved_remove_background
except ImportError:
    from effects import apply_effects
    from filters import apply_advanced_edits, apply_basic_edits, apply_intermediate_edits, apply_misc
    from image_utils import _apply_crop, _apply_pixelate, image_to_ascii, improved_remove_background


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
    img = apply_basic_edits(img, grayscale, brightness, contrast, flip, blur)
    img = apply_intermediate_edits(img, sharpness, saturation)
    img = apply_advanced_edits(img, sepia, edge_detect, cartoon, glitch, invert, emboss)
    img = apply_effects(
        img, apply_kaleidoscope, segments, apply_wave, wave_amplitude, wave_length,
        channel_swap_type, apply_mosaic, tile_size, apply_noise, noise_level,
        apply_vignette, vignette_intensity
    )
    img = apply_misc(
        img, rotation, opacity, text_overlay, text_color, text_font_size,
        text_position, font_type, make_gif, gif_frames
    )

    # Placeholder for ASCII output
    ascii_text = ""

    return img, ascii_text


def join_images(image1, image2, join_direction="horizontal"):
    """Join two images horizontally or vertically."""
    if image1 is None or image2 is None:
        return None

    from PIL import Image

    # Convert to RGB if needed
    if image1.mode != 'RGB':
        image1 = image1.convert('RGB')
    if image2.mode != 'RGB':
        image2 = image2.convert('RGB')

    if join_direction == "horizontal":
        # Join horizontally - images side by side
        # Get the maximum height and resize both images to match
        max_height = max(image1.height, image2.height)

        # Resize images to have the same height
        img1_resized = image1.resize((int(image1.width * max_height / image1.height), max_height),
         Image.Resampling.LANCZOS)
        img2_resized = image2.resize((int(image2.width * max_height / image2.height), max_height),
         Image.Resampling.LANCZOS)

        # Create new image with combined width
        joined_width = img1_resized.width + img2_resized.width
        joined_img = Image.new('RGB', (joined_width, max_height))

        # Paste images side by side
        joined_img.paste(img1_resized, (0, 0))
        joined_img.paste(img2_resized, (img1_resized.width, 0))

    else:  # vertical
        # Join vertically - images stacked
        # Get the maximum width and resize both images to match
        max_width = max(image1.width, image2.width)

        # Resize images to have the same width
        img1_resized = image1.resize((max_width, int(image1.height * max_width / image1.width)),
         Image.Resampling.LANCZOS)
        img2_resized = image2.resize((max_width, int(image2.height * max_width / image2.width)),
         Image.Resampling.LANCZOS)

        # Create new image with combined height
        joined_height = img1_resized.height + img2_resized.height
        joined_img = Image.new('RGB', (max_width, joined_height))

        # Paste images vertically
        joined_img.paste(img1_resized, (0, 0))
        joined_img.paste(img2_resized, (0, img1_resized.height))

    return joined_img


def generate_ascii(original, *params):
    """Generate ASCII art from edited image."""
    img = edit_image(original, *params)[0]
    if img is None:
        return ""
    return image_to_ascii(img)
