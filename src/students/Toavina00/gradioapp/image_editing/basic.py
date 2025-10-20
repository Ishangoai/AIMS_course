import numpy as np
from scipy import ndimage


def img_to_grayscale(image: np.ndarray):
    if len(image.shape) == 3:
        image = image.dot([0.2989, 0.5870, 0.1140])
    return image


def invert_color(image: np.ndarray):
    image *= -1
    return image


def flip_vertical(image: np.ndarray):
    n = image.shape[1]
    if len(image.shape) == 2:
        image = image[:, n::-1]
    else:
        image = image[:, n::-1, :]
    return image


def flip_horizontal(image: np.ndarray):
    n = image.shape[0]
    if len(image.shape) == 2:
        image = image[n::-1, :]
    else:
        image = image[n::-1, :, :]
    return image


def change_brightness(image: np.ndarray, value: float):
    image *= value
    image = np.clip(image, -1.0, 1.0)
    return image


def change_contrast(image: np.ndarray, value: float):
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    image = (image - mean) * value + mean
    image = np.clip(image, -1.0, 1.0)
    return image


def rotate_image(image: np.ndarray, radius: float):
    image = ndimage.rotate(image, radius, reshape=True)
    return image


def blur_image(image: np.ndarray, sigma: float):
    if len(image.shape) == 2:
        return ndimage.gaussian_filter(image, sigma)
    else:
        return ndimage.gaussian_filter(image, (sigma, sigma, 0))


def normalize_image(image: np.ndarray):
    image = image.astype(np.float64)
    image -= 255.0 / 2
    image /= 255.0 / 2
    return image


def denormalize_image(image: np.ndarray):
    image *= 255.0 / 2
    image += 255.0 / 2
    image = np.clip(image, 0, 255)
    image = image.astype(int)
    return image


def transform_image(
    image: np.ndarray | None,
    to_grayscale: bool,
    invert: bool,
    flip_h: bool,
    flip_v: bool,
    brightness: float,
    contrast: float,
    rotate: float,
    blur_sigma: float,
):
    if image is None:
        return None

    image = normalize_image(image)  # Normalize image to pixels of (-1.0, 1.0) for numerical stability
    image = change_brightness(image, brightness)
    image = change_contrast(image, contrast)
    image = blur_image(image, blur_sigma)

    if to_grayscale:
        image = img_to_grayscale(image)

    if flip_h:
        image = flip_horizontal(image)

    if flip_v:
        image = flip_vertical(image)

    if invert:
        image = invert_color(image)

    image = rotate_image(image, rotate)
    image = denormalize_image(image)  # Restore image to pixels of (0, 255)
    return image
