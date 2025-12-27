# -----------------------------------------------------------------------------
# Background Removal with AI
# Author: Mohammad Reza Niknam
# Institution: Shahrood University of Technology
# Program: Master's in Telecommunications Systems
# Description: This code implements a personalized background removal tool
#              using artificial intelligence. All rights reserved.
# -----------------------------------------------------------------------------

# Import required libraries
import sys
import time
import onnxruntime as ort  # Library to run ONNX models
import cv2  # OpenCV library for image processing
import numpy as np  # Library for numerical operations

# Import custom modules
sys.path.append('remove_background_company/utils') # Add utility modules to system path if needed

# Logger setup
from logging import getLogger
import webcamera_utils  # Utilities for handling webcam input/output
from image_utils import normalize_image  # Utility for image normalization
from detector_utils import load_image  # Utility to load images
from arg_utils import get_base_parser, get_savepath, update_parser  # Argument handling utilities

logger = getLogger(__name__)  # Initialize logger

# ======================
# Parameters 1
# ======================
IMAGE_PATH = 'chair.jpg'  # Default path for input image
SAVE_IMAGE_PATH = 'output_chair.png'  # Default path for saving output image
IMAGE_SIZE = 1024  # Default input image size for the model

# ======================
# Argument Parser Config
# ======================
# Initialize argument parser with default parameters
parser = get_base_parser(
    'DIS segmentation model', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '--img-size', type=int, default=IMAGE_SIZE,
    help='Input image size for the model'
)
parser.add_argument(
    '-n', '--normal',
    action='store_true',
    help='Use non-optimized model if specified, otherwise use optimized model by default'
)
args = update_parser(parser)  # Update parser with arguments

# ======================
# Parameters 2
# ======================
# Set model weight file based on selected mode (normal or optimized)
WEIGHT_PATH = 'isnet-general-use.onnx' if not args.normal else 'dis.onnx'
MODEL_PATH = WEIGHT_PATH + '.prototxt'  # Model configuration file

# ======================
# Utils
# ======================

def preprocess(img):
    """Preprocess input image for model prediction."""
    im_h, im_w, _ = img.shape  # Get original image dimensions
    s = args.img_size  # Set target size from arguments

    # Resize and normalize image
    img = cv2.resize(img, (s, s), interpolation=cv2.INTER_LINEAR)
    img = normalize_image(img, normalize_type='127.5') / 2

    # Reorder dimensions to match model input (HWC -> CHW)
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img.astype(np.float32)  # Convert to float32 type

    return img

def predict(session, img):
    """Run inference on the input image and return the prediction mask."""
    im_h, im_w = img.shape[:2]  # Original image dimensions
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB format
    img = preprocess(img)  # Preprocess image for model

    # Perform inference
    input_name = session.get_inputs()[0].name  # Get input name
    output = session.run(None, {input_name: img})  # Run model prediction
    pred = output[0][0]  # Get the prediction output

    # Reshape and normalize prediction mask
    mask = pred.transpose(1, 2, 0)  # Convert CHW -> HWC
    mask = cv2.resize(mask, (im_w, im_h), interpolation=cv2.INTER_LINEAR)[:, :, np.newaxis]
    ma = np.max(mask)  # Max value in mask for normalization
    mi = np.min(mask)  # Min value in mask for normalization
    mask = (mask - mi) / (ma - mi)  # Normalize mask values to [0, 1]

    return mask

# ======================
# Main functions
# ======================
def recognize_from_image():
    """Run inference on an image and save the output."""
    # Initialize model session
    session = ort.InferenceSession(WEIGHT_PATH)

    # Loop over input images specified in args
    for image_path in args.input:
        logger.info(image_path)  # Log image path
        img = load_image(image_path)  # Load image
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # Convert to BGR format

        # Perform inference
        logger.info('Start inference...')
        if args.benchmark:  # If benchmarking
            logger.info('BENCHMARK mode')
            for i in range(5):  # Run multiple inference rounds for timing
                start = int(round(time.time() * 1000))
                mask = predict(session, img)  # Generate mask
                end = int(round(time.time() * 1000))
                logger.info(f'\tProcessing time {end - start} ms')
        else:
            mask = predict(session, img)

        # Combine mask and original image for output
        res_img = np.concatenate((mask * img, mask * 255), axis=2).astype(np.uint8)

        # Save result
        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)  # Write result image to file

    logger.info('Script finished successfully.')

def recognize_from_video():
    """Run inference on video frames and display/save output."""
    session = ort.InferenceSession(WEIGHT_PATH)  # Initialize model session
    flag_set_shape = False  # Flag for frame shape

    capture = webcamera_utils.get_capture(args.video)  # Open video capture

    # Initialize video writer if saving to a video file
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        save_h, save_w = webcamera_utils.calc_adjust_fsize(
            f_h, f_w, IMAGE_HEIGHT, IMAGE_WIDTH
        )
        writer = webcamera_utils.get_writer(args.savepath, save_h, save_w)
    else:
        writer = None

    frame_shown = False  # Track if frame is displayed
    while True:
        ret, frame = capture.read()  # Capture frame
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:  # Exit on 'q' key
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # Perform inference
        mask = predict(session, frame)

        # Display result
        res_img = (mask * frame).astype(np.uint8)  # Apply mask to frame
        cv2.imshow('frame', res_img)
        frame_shown = True

        # Save result if writer is available
        if writer is not None:
            writer.write(res_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')

def main():
    """Main function to handle image or video mode based on arguments."""
    if args.video is not None:
        # Video mode
        recognize_from_video()
    else:
        # Image mode
        recognize_from_image()

# Entry point
if __name__ == '__main__':
    main()
