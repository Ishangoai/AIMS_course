import numpy as np
from gradioapp.image_editing.basic import denormalize_image, img_to_grayscale, normalize_image
from scipy import ndimage


def _check_bound(x: int, y: int, x_max: int, y_max: int):
    return (0 <= x < x_max) and (0 <= y < y_max)


def _dithering(image: np.ndarray):
    m, n = image.shape[:2]

    image = normalize_image(image)

    # Floyd–Steinberg dithering Algorithm
    for x in range(m):
        for y in range(n):
            oldpixel = image[x, y]
            newpixel = np.where(oldpixel > 0, 1.0, -1.0)
            image[x, y] = newpixel
            quant_error = oldpixel - newpixel

            if _check_bound(x, y + 1, m, n):
                image[x, y + 1] += quant_error * 5 / 16
            if _check_bound(x + 1, y, m, n):
                image[x + 1, y] += quant_error * 7 / 16
            if _check_bound(x + 1, y + 1, m, n):
                image[x + 1, y + 1] += quant_error * 1 / 16
            if _check_bound(x - 1, y + 1, m, n):
                image[x - 1, y + 1] += quant_error * 3 / 16

    image = denormalize_image(image)

    return image


def color_dithering(image: np.ndarray):
    return _dithering(image)


def gray_dithering(image: np.ndarray):
    image = img_to_grayscale(image)
    return _dithering(image)


def sobel_h_filter(image: np.ndarray):
    return ndimage.sobel(image, 0)


def sobel_v_filter(image: np.ndarray):
    return ndimage.sobel(image, 1)


def sobel_magnitude(image: np.ndarray):
    sobel_h = ndimage.sobel(image, 0)  # horizontal gradient
    sobel_v = ndimage.sobel(image, 1)  # vertical gradient
    magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
    magnitude *= 255.0 / np.max(magnitude)  # normalization
    magnitude = magnitude.clip(0, 255).astype(int)
    return magnitude


def laplace_filter(image: np.ndarray):
    return ndimage.laplace(image)


def _get_neighbour(i_x: int, i_y: int, grad_ang: float):
    if grad_ang <= 22.5:
        neighb_1_x, neighb_1_y = i_x - 1, i_y
        neighb_2_x, neighb_2_y = i_x + 1, i_y
    elif grad_ang > 22.5 and grad_ang <= 67.5:
        neighb_1_x, neighb_1_y = i_x - 1, i_y - 1
        neighb_2_x, neighb_2_y = i_x + 1, i_y + 1
    elif grad_ang > 67.5 and grad_ang <= 112.5:
        neighb_1_x, neighb_1_y = i_x, i_y - 1
        neighb_2_x, neighb_2_y = i_x, i_y + 1
    elif grad_ang > 112.5 and grad_ang <= 157.5:
        neighb_1_x, neighb_1_y = i_x - 1, i_y + 1
        neighb_2_x, neighb_2_y = i_x + 1, i_y - 1
    else:
        neighb_1_x, neighb_1_y = i_x - 1, i_y
        neighb_2_x, neighb_2_y = i_x + 1, i_y

    return neighb_1_x, neighb_1_y, neighb_2_x, neighb_2_y


def _non_max_suppression(ang: np.ndarray, mag: np.ndarray, width: int, height: int):
    for i_x in range(width):
        for i_y in range(height):
            grad_ang = ang[i_y, i_x]
            grad_ang = abs(grad_ang - 180) if abs(grad_ang) > 180 else abs(grad_ang)

            neighb_1_x, neighb_1_y, neighb_2_x, neighb_2_y = _get_neighbour(i_x, i_y, grad_ang)

            if 0 <= neighb_1_x < width and 0 <= neighb_1_y < height:
                if mag[i_y, i_x] < mag[neighb_1_y, neighb_1_x]:
                    mag[i_y, i_x] = 0
                    continue

            if 0 <= neighb_2_x < width and 0 <= neighb_2_y < height:
                if mag[i_y, i_x] < mag[neighb_2_y, neighb_2_x]:
                    mag[i_y, i_x] = 0

    return ang, mag


def _thresholding(mag: np.ndarray, weak_th: float, strong_th: float, width: int, height: int):
    ids = np.zeros((height, width))

    for i_x in range(width):
        for i_y in range(height):
            grad_mag = mag[i_y, i_x]
            if grad_mag < weak_th:
                mag[i_y, i_x] = 0
            elif strong_th > grad_mag >= weak_th:
                ids[i_y, i_x] = 1
            else:
                ids[i_y, i_x] = 2
    return mag


def canny_detector(image: np.ndarray):
    image = img_to_grayscale(image)

    image = ndimage.gaussian_filter(image, sigma=1.5)

    gx = sobel_h_filter(image)
    gy = sobel_v_filter(image)
    mag = np.hypot(gx, gy)
    ang = np.arctan2(gy, gx)

    height, width = image.shape[:2]

    mag_max = np.max(mag)
    weak_th = mag_max * 0.1
    strong_th = mag_max * 0.5

    ang, mag = _non_max_suppression(ang, mag, width, height)
    mag = _thresholding(mag, weak_th, strong_th, width, height)
    mag = mag.clip(0, 255).astype(int)

    return mag


def histogram_equalizer(image: np.ndarray):
    hist, _ = np.histogram(image.flatten(), 256, (0, 256))

    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype(int)
    image = cdf[image]

    return image
