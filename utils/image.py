import os
import sys
from logging import getLogger

import cv2
import numpy as np

# Set up logging
logger = getLogger(__name__)

def imread(filename, flags=cv2.IMREAD_COLOR):
    """
    Reads an image from a file and decodes it using OpenCV.

    Parameters
    ----------
    filename: str
        The path of the file to read.
    flags: int, default=cv2.IMREAD_COLOR
        Specifies the color type of the loaded image.

    Returns
    -------
    img: numpy array
        Decoded image in numpy array format.
    """
    if not os.path.isfile(filename):
        # Log an error and exit if the file does not exist
        logger.error(f"File does not exist: {filename}")
        sys.exit()
    data = np.fromfile(filename, np.int8)  # Read file as a numpy array
    img = cv2.imdecode(data, flags)  # Decode the image data
    return img

def normalize_image(image, normalize_type='255'):
    """
    Normalize an image based on the specified method.

    Parameters
    ----------
    image: numpy array
        The image to normalize.
    normalize_type: str, default='255'
        Type of normalization:
        - '255': Scales to [0, 1] by dividing by 255.0.
        - '127.5': Scales to [-1, 1] by dividing by 127.5 and subtracting 1.
        - 'ImageNet': Uses ImageNet mean and std for normalization.
        - 'None': No normalization.

    Returns
    -------
    normalized_image: numpy array
        The normalized image.
    """
    if normalize_type == 'None':
        return image
    elif normalize_type == '255':
        return image / 255.0
    elif normalize_type == '127.5':
        return image / 127.5 - 1.0
    elif normalize_type == 'ImageNet':
        # ImageNet normalization using mean and std
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image / 255.0
        for i in range(3):  # Normalize each channel separately
            image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]
        return image
    else:
        # Log an error and exit if an unknown normalization type is given
        logger.error(f'Unknown normalize_type is given: {normalize_type}')
        sys.exit()

def load_image(
        image_path,
        image_shape,
        rgb=True,
        normalize_type='255'
):
    """
    Loads and preprocesses an image.

    Parameters
    ----------
    image_path: str
        Path of the image to load.
    image_shape: (int, int)
        Size (height, width) to resize the image.
    rgb: bool, default=True
        Load the image as RGB if True, grayscale if False.
    normalize_type: str, default='255'
        Type of normalization (see normalize_image for options).

    Returns
    -------
    image: numpy array
        Preprocessed image ready for model input.
    """
    image = imread(image_path, int(rgb))  # Load image with specified color flag
    if rgb:
        # Convert image to RGB format if specified
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = normalize_image(image, normalize_type)  # Normalize image
    # Resize the image to match model's expected input shape
    image = cv2.resize(image, (image_shape[1], image_shape[0]))
    return image

def get_image_shape(image_path):
    """
    Retrieves the dimensions of the image at the specified path.

    Parameters
    ----------
    image_path: str
        Path to the image file.

    Returns
    -------
    (height, width): tuple of int
        Dimensions of the image.
    """
    tmp = imread(image_path)
    height, width = tmp.shape[0], tmp.shape[1]
    return height, width

def draw_texts(img, texts, font_scale=0.7, thickness=2):
    """
    Draws multiple lines of text on an image.

    Parameters
    ----------
    img: numpy array
        Image on which to draw text.
    texts: str or list of str
        Text to display, one string or a list of strings for multiple lines.
    font_scale: float, default=0.7
        Scale of the font.
    thickness: int, default=2
        Thickness of the text.

    """
    h, w, c = img.shape
    offset_x = 10
    initial_y = 0
    dy = int(img.shape[1] / 15)  # Spacing between lines of text
    color = (0, 0, 0)  # Text color (black)

    texts = [texts] if isinstance(texts, str) else texts

    for i, text in enumerate(texts):
        # Calculate y-offset for each line of text
        offset_y = initial_y + (i + 1) * dy
        cv2.putText(img, text, (offset_x, offset_y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, cv2.LINE_AA)

def draw_result_on_img(img, texts, w_ratio=0.35, h_ratio=0.2, alpha=0.4):
    """
    Draws a semi-transparent overlay with text on the image.

    Parameters
    ----------
    img: numpy array
        Image on which to draw the overlay and text.
    texts: str or list of str
        Text to display on the overlay.
    w_ratio: float, default=0.35
        Width of the overlay as a fraction of image width.
    h_ratio: float, default=0.2
        Height of the overlay as a fraction of image height.
    alpha: float, default=0.4
        Transparency level of the overlay.

    Returns
    -------
    mat_img: numpy array
        Image with overlay and text drawn on it.
    """
    overlay = img.copy()  # Copy the image to draw overlay
    pt1 = (0, 0)
    pt2 = (int(img.shape[1] * w_ratio), int(img.shape[0] * h_ratio))

    mat_color = (200, 200, 200)  # Overlay color (light gray)
    fill = -1  # Fill the rectangle
    cv2.rectangle(overlay, pt1, pt2, mat_color, fill)

    # Combine original image and overlay with transparency
    mat_img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # Draw the specified texts on the overlay image
    draw_texts(mat_img, texts)
    return mat_img
