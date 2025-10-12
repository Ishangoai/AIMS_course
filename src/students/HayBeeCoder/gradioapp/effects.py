import random

import numpy as np
from PIL import Image
from scipy import ndimage


def kaleidoscope(img, segments=4):
    """Apply kaleidoscope effect to image."""
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
    """Apply wave distortion effect to image."""
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
    """Swap color channels in image."""
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
    """Apply mosaic effect to image."""
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
    """Add random noise to image."""
    img = img.convert("RGB")
    img_arr = np.array(img)
    noise = np.random.normal(0, noise_level * 255, img_arr.shape)
    noisy_img = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)


def vignette(img, intensity=0.5):
    """Apply vignette effect to image."""
    img = img.convert("RGB")
    w, h = img.size
    img_arr = np.array(img)
    x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    mask = 1 - intensity * (x**2 + y**2)
    mask = np.clip(mask, 0, 1)[:, :, np.newaxis]
    result = (img_arr * mask).astype(np.uint8)
    return Image.fromarray(result)


def apply_effects(img, apply_kaleidoscope, segments, apply_wave, wave_amplitude, wave_length,
                  channel_swap_type, apply_mosaic, tile_size, apply_noise, noise_level,
                  apply_vignette, vignette_intensity):
    """Apply all effects to image."""
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
