try:
    from .effects import apply_effects
    from .filters import apply_advanced_edits, apply_basic_edits, apply_intermediate_edits, apply_misc
    from .image_utils import _apply_crop, _apply_pixelate, improved_remove_background, image_to_ascii
except ImportError:
    from effects import apply_effects
    from filters import apply_advanced_edits, apply_basic_edits, apply_intermediate_edits, apply_misc
    from image_utils import _apply_crop, _apply_pixelate, improved_remove_background, image_to_ascii


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


def generate_ascii(original, *params):
    """Generate ASCII art from edited image."""
    img = edit_image(original, *params)[0]
    if img is None:
        return ""
    return image_to_ascii(img)
