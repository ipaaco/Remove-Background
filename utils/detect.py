import os
import sys
from logging import getLogger

import onnxruntime as ort  # Replacing ailia with onnxruntime
import cv2
import numpy as np
import json

logger = getLogger(__name__)

sys.path.append(os.path.dirname(__file__))
from image_utils import imread  # Import utility to read images


def preprocessing_img(img):
    """
    Convert image to BGRA format if needed.
    
    Parameters
    ----------
    img : numpy array
        Input image
    
    Returns
    -------
    img : numpy array
        Image in BGRA format
    """
    if len(img.shape) < 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    return img


def load_image(image_path):
    """
    Load image from file and preprocess it.
    
    Parameters
    ----------
    image_path : str
        Path to image file
    
    Returns
    -------
    img : numpy array
        Preprocessed image
    """
    if os.path.isfile(image_path):
        img = imread(image_path, cv2.IMREAD_UNCHANGED)
    else:
        logger.error(f'{image_path} not found.')
        sys.exit()
    return preprocessing_img(img)


def hsv_to_rgb(h, s, v):
    """
    Convert HSV values to RGB for color display.
    
    Parameters
    ----------
    h, s, v : int
        Hue, saturation, and value components in HSV color space.
        
    Returns
    -------
    color : tuple
        RGB color as a tuple with alpha channel.
    """
    bgr = cv2.cvtColor(
        np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]), 255)


def letterbox_convert(frame, det_shape):
    """
    Adjust the size of the frame to fit the model input shape.
    
    Parameters
    ----------
    frame : numpy array
        Original frame from webcam
    det_shape : tuple
        Target detection shape (height, width)
        
    Returns
    -------
    resized_img : numpy array
        Resized image to fit the model input shape
    """
    height, width = det_shape[0], det_shape[1]
    f_height, f_width = frame.shape[0], frame.shape[1]
    scale = np.max((f_height / height, f_width / width))

    # Prepare padded image with target shape
    img = np.zeros(
        (int(round(scale * height)), int(round(scale * width)), 3),
        np.uint8
    )
    start = (np.array(img.shape) - np.array(frame.shape)) // 2
    img[start[0]: start[0] + f_height, start[1]: start[1] + f_width] = frame
    resized_img = cv2.resize(img, (width, height))
    return resized_img


def reverse_letterbox(detections, img, det_shape):
    """
    Revert the letterbox effect for detected bounding boxes.
    
    Parameters
    ----------
    detections : list
        List of detection objects with normalized coordinates.
    img : numpy array
        Original input image.
    det_shape : tuple
        Detection shape (height, width).
        
    Returns
    -------
    new_detections : list
        List of adjusted detections with absolute coordinates.
    """
    h, w = img.shape[0], img.shape[1]
    scale = max(h / det_shape[0], w / det_shape[1])
    start = (det_shape[0:2] - np.array(img.shape[0:2]) / scale) // 2
    pad_x, pad_y = start[1] * scale, start[0] * scale

    new_detections = []
    for detection in detections:
        adjusted_detection = {
            'category': detection['category'],
            'prob': detection['prob'],
            'x': (detection['x'] * (w + pad_x * 2) - pad_x) / w,
            'y': (detection['y'] * (h + pad_y * 2) - pad_y) / h,
            'w': detection['w'] * (w + pad_x * 2) / w,
            'h': detection['h'] * (h + pad_y * 2) / h
        }
        new_detections.append(adjusted_detection)

    return new_detections


def plot_results(detections, img, category=None, segm_masks=None, logging=True):
    """
    Draw bounding boxes and labels on the image.
    
    Parameters
    ----------
    detections : list
        List of detection objects.
    img : numpy array
        Image on which to draw the results.
    category : list
        List of category names.
    segm_masks : list
        List of segmentation masks (optional).
    logging : bool
        Whether to log the detection results.
    
    Returns
    -------
    img : numpy array
        Image with plotted results.
    """
    h, w = img.shape[0], img.shape[1]
    count = len(detections)

    if logging:
        print(f'object_count={count}')

    colors = []
    for idx in range(count):
        obj = detections[idx]
        category_name = category[int(obj['category'])] if category is not None else obj['category']
        
        if logging:
            print(f'+ idx={idx}')
            print(f'  category={obj["category"]}[ {category_name} ]')
            print(f'  prob={obj["prob"]}')
            print(f'  x={obj["x"]}')
            print(f'  y={obj["y"]}')
            print(f'  w={obj["w"]}')
            print(f'  h={obj["h"]}')

        color = hsv_to_rgb(256 * idx / (count + 1), 255, 255)
        colors.append(color)

    # Draw segmentation masks if available
    if segm_masks:
        for idx in range(count):
            mask = segm_masks[idx].astype(bool)
            img[mask] = img[mask] * 0.7 + np.array(colors[idx][:3]) * 0.3

    # Draw bounding boxes and labels
    for idx, obj in enumerate(detections):
        top_left = (int(w * obj['x']), int(h * obj['y']))
        bottom_right = (int(w * (obj['x'] + obj['w'])), int(h * (obj['y'] + obj['h'])))
        cv2.rectangle(img, top_left, bottom_right, colors[idx], 4)

        # Draw label
        text = f"{category[int(obj['category'])]} {obj['prob']:.2f}"
        cv2.putText(
            img, text, (top_left[0], top_left[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
    return img


def write_predictions(file_name, detections, img=None, category=None, file_type='txt'):
    """
    Write detection results to a file in JSON or TXT format.
    
    Parameters
    ----------
    file_name : str
        Output file path.
    detections : list
        List of detection objects.
    img : numpy array
        Original image, used to calculate absolute coordinates.
    category : list
        List of category names.
    file_type : str
        File format, either 'json' or 'txt'.
    """
    h, w = (img.shape[0], img.shape[1]) if img is not None else (1, 1)

    if file_type == 'json':
        results = [
            {
                'category': category[int(obj['category'])] if category else obj['category'],
                'prob': obj['prob'],
                'x': obj['x'] * w,
                'y': obj['y'] * h,
                'w': obj['w'] * w,
                'h': obj['h'] * h
            } for obj in detections
        ]
        with open(file_name, 'w') as f:
            json.dump(results, f, indent=2)
    else:
        with open(file_name, 'w') as f:
            for obj in detections:
                label = category[int(obj['category'])] if category else obj['category']
                f.write(f"{label} {obj['prob']} {int(obj['x'] * w)} {int(obj['y'] * h)} "
                        f"{int(obj['w'] * w)} {int(obj['h'] * h)}\n")
