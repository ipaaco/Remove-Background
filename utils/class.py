import os
import sys
import numpy as np
import cv2
import json

# Constants for controlling the number of top classes and display settings
MAX_CLASS_COUNT = 3
RECT_WIDTH = 640
RECT_HEIGHT = 20
RECT_MARGIN = 2

def get_top_scores(classifier, top_k=MAX_CLASS_COUNT):
    """
    Retrieve the top K classification scores.
    
    Parameters
    ----------
    classifier : list or classifier object
        The classifier outputs, either a structured object or array of probabilities.
    top_k : int
        The number of top classes to retrieve.
        
    Returns
    -------
    top_scores : list
        List of top K class indices.
    scores : dict or list
        Mapping of class indices to probabilities.
    """
    if hasattr(classifier, 'get_class_count'):
        # Handle structured classifier API (like `ailia` API)
        count = classifier.get_class_count()
        scores = {}
        top_scores = []
        for idx in range(count):
            obj = classifier.get_class(idx)
            top_scores.append(obj.category)  # Store top class categories
            scores[obj.category] = obj.prob  # Store probability scores
    else:
        # Handle generic array-based API, such as ONNX model output
        classifier = classifier[0]  # Select main output if in list form
        top_scores = classifier.argsort()[-top_k:][::-1]  # Get top K indices
        scores = classifier  # Use classifier directly as score list
    return top_scores, scores


def print_results(classifier, labels, top_k=MAX_CLASS_COUNT):
    """
    Print the classification results with top class labels and probabilities.
    
    Parameters
    ----------
    classifier : list or classifier object
        The classifier outputs, either a structured object or array of probabilities.
    labels : list
        List of class labels.
    top_k : int
        The number of top classes to display.
    """
    top_scores, scores = get_top_scores(classifier, top_k)
    top_k = min(len(top_scores), top_k)

    print('==============================================================')
    print(f'class_count={top_k}')
    for idx in range(top_k):
        print(f'+ idx={idx}')
        print(f'  category={top_scores[idx]}[{labels[top_scores[idx]]}]')
        print(f'  prob={scores[top_scores[idx]]}')


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


def plot_results(input_image, classifier, labels, top_k=MAX_CLASS_COUNT, logging=True):
    """
    Plot classification results on the input image with colored rectangles.
    
    Parameters
    ----------
    input_image : numpy.ndarray
        The image to draw the results on.
    classifier : list or classifier object
        The classifier outputs, either a structured object or array of probabilities.
    labels : list
        List of class labels.
    top_k : int
        The number of top classes to plot.
    logging : bool
        If True, prints results to console.
    """
    x = RECT_MARGIN
    y = RECT_MARGIN
    w = RECT_WIDTH
    h = RECT_HEIGHT

    top_scores, scores = get_top_scores(classifier, top_k)
    top_k = min(len(top_scores), top_k)

    if logging:
        print('==============================================================')
        print(f'class_count={top_k}')
    for idx in range(top_k):
        if logging:
            print(f'+ idx={idx}')
            print(f'  category={top_scores[idx]}[{labels[top_scores[idx]]}]')
            print(f'  prob={scores[top_scores[idx]]}')
        
        # Text overlay with class label and probability
        text = f'category={top_scores[idx]}[{labels[top_scores[idx]]}] prob={scores[top_scores[idx]]}'

        # Assign color based on HSV for better visualization
        color = hsv_to_rgb(256 * top_scores[idx] / (len(labels) + 1), 128, 255)

        # Draw rectangle and add text for each class
        cv2.rectangle(input_image, (x, y), (x + w, y + h), color, thickness=-1)
        text_position = (x + 4, y + int(RECT_HEIGHT / 2) + 4)
        color = (0, 0, 0)  # Text color
        fontScale = 0.5

        cv2.putText(
            input_image,
            text,
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale,
            color,
            1
        )

        y = y + h + RECT_MARGIN  # Move to next rectangle position


def write_predictions(file_name, classifier, labels, file_type='txt'):
    """
    Write classification predictions to a file in JSON or TXT format.
    
    Parameters
    ----------
    file_name : str
        The name of the output file.
    classifier : list or classifier object
        The classifier outputs, either a structured object or array of probabilities.
    labels : list
        List of class labels.
    file_type : str
        Output file format: 'json' or 'txt'.
    """
    if file_type == 'json':
        top_k = MAX_CLASS_COUNT
        top_scores, scores = get_top_scores(classifier, top_k)
        top_k = min(len(top_scores), top_k)
        out_data = []
        for idx in range(top_k):
            out_data.append({
                'idx': idx,
                'category': int(top_scores[idx]),
                'label': labels[top_scores[idx]],
                'prob': float(scores[top_scores[idx]])
            })
        # Write to JSON with pretty formatting
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(out_data, f, indent=2, ensure_ascii=False)
    else:
        top_k = 5
        top_scores, scores = get_top_scores(classifier, top_k)
        top_k = min(len(top_scores), top_k)
        # Write to TXT file with each line for a label and its probability
        with open(file_name, 'w', encoding='utf-8') as f:
            for idx in range(top_k):
                f.write('%s %d %f\n' % (
                    labels[top_scores[idx]].replace(' ', '_'),
                    top_scores[idx],
                    scores[top_scores[idx]]
                ))
