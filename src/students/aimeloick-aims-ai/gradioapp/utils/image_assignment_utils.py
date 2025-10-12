import os
import tempfile
import zipfile

from PIL import Image, ImageEnhance, ImageOps


# ==============================
# Image Processing Functions
# ==============================
def grayscale(image):
    """Convert image to grayscale."""
    return ImageOps.grayscale(image).convert("RGB")


def brightness(image, factor):
    """Adjust image brightness."""
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def contrast(image, factor):
    """Adjust image contrast."""
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def rotate(image, degree):
    """Rotate image by given degree."""
    return image.rotate(degree, expand=True)


def remove_white_background(image, threshold):
    """Make white background transparent based on threshold."""
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


def transform_image(image, grayscale_or_not, brightness_factor, contrast_factor, degree, threshold, remove_bg):
    """Apply all transformations to an image."""
    # Step 1: Grayscale
    if grayscale_or_not == "Grayscale":
        image_gray = grayscale(image)
    else:
        image_gray = image

    # Step 2: Brightness
    brightened_image = brightness(image_gray, brightness_factor)

    # Step 3: Contrast
    contrasted_image = contrast(brightened_image, contrast_factor)

    # Step 4: Rotation
    rotated_image = rotate(contrasted_image, degree)

    # Step 5: Remove background (optional)
    if remove_bg:
        final_image = remove_white_background(rotated_image, threshold)
    else:
        final_image = rotated_image

    return final_image


# ==============================
# Single Image Functions
# ==============================
def transform_single_image(image, grayscale_or_not, brightness_val, contrast_val, degree, threshold, remove_bg):
    """Apply transformations to a single image."""
    if image is None:
        return None

    try:
        result = transform_image(image, grayscale_or_not, brightness_val, contrast_val, degree, threshold, remove_bg)
        return result
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return image


def download_single_image(image):
    """Save single image to temporary file for download."""
    if image is None:
        return None

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image.save(temp_file.name)
    temp_file.close()
    return temp_file.name


# ==============================
# Batch Processing Functions
# ==============================
def transform_folder(files, grayscale_or_not, brightness_val, contrast_val, degree, threshold, remove_bg):
    """Transform all images in uploaded folder and return a ZIP file."""
    if not files:
        return "No files uploaded", None

    temp_dir = tempfile.mkdtemp()
    output_dir = os.path.join(temp_dir, "transformed_images")
    os.makedirs(output_dir, exist_ok=True)

    processed_count = 0
    failed_count = 0

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

    try:
        for file_obj in files:
            file_path = file_obj.name
            file_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_name)[1].lower()

            if file_ext not in image_extensions:
                continue

            try:
                image = Image.open(file_path)
                transformed_image = transform_image(
                    image, grayscale_or_not, brightness_val, contrast_val,
                    degree, threshold, remove_bg
                )

                output_path = os.path.join(output_dir, file_name)

                if file_ext in ['.jpg', '.jpeg'] and transformed_image.mode == 'RGBA':
                    rgb_image = Image.new('RGB', transformed_image.size, (255, 255, 255))
                    rgb_image.paste(transformed_image, mask=transformed_image.split()[3])
                    rgb_image.save(output_path, quality=95)
                else:
                    transformed_image.save(output_path)

                processed_count += 1

            except Exception as e:
                failed_count += 1
                print(f"Failed to process {file_name}: {e}")
                continue

        if processed_count == 0:
            return "No images could be processed", None

        zip_path = os.path.join(temp_dir, "transformed_images.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files_in_dir in os.walk(output_dir):
                for file in files_in_dir:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, output_dir)
                    zipf.write(file_path, arcname)

        status_msg = f"Successfully processed {processed_count} images!"
        if failed_count > 0:
            status_msg += f"\n{failed_count} images failed"

        return status_msg, zip_path

    except Exception as e:
        return f"Error: {str(e)}", None
