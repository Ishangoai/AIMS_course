# scripts/utils.py
import os
import numpy as np
from PIL import Image, ImageOps, ImageEnhance

def grayscale(image):
    #applying grayscale method 
    image_gray = ImageOps.grayscale(image)
    return image_gray

def brightness(image, brightness_factor):
    enhancer = ImageEnhance.Brightness(image)
    brightened_image = enhancer.enhance(brightness_factor)
    return brightened_image

def contrast(image, contrast_factor):
    enhancer = ImageEnhance.Contrast(image)
    contrasted_image = enhancer.enhance(contrast_factor)
    return contrasted_image

def rotate_image(image, degree):
    return image.rotate(degree, expand = True)

   
def apply_zoom(image, zoom_factor):
    """Zoom in/out l'image par un facteur."""
    if zoom_factor == 1.0:
        return image
    w, h = image.size
    new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
    return image.resize((new_w, new_h), resample =  Image.LANCZOS)

def remove_white_background(image, threshold):
    """Rend le fond blanc transparent."""
    img = image.convert("RGBA")
    datas = img.getdata()

    new_data = []
    for item in datas:
        if item[0] > threshold and item[1] > threshold and item[2] > threshold:
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)
    img.putdata(new_data)
    return img

def transform_image(image, grayscale_or_not, brightness_factor, contrast_factor, degree, threshold):
    if grayscale_or_not == "Grayscale":
         image_gray = grayscale(image)
    else :
         image_gray = image
    brightened_image = brightness(image_gray, brightness_factor)
    contrastn_image = contrast(brightened_image, contrast_factor)
    rotation_image = rotate_image(contrastn_image, degree)
    without_background_image = remove_white_background(rotation_image, threshold)
    #zoom_image = apply_zoom(without_background_image, zoom_factor)
    return without_background_image 


