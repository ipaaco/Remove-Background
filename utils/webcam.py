import os
import sys
import re

import numpy as np
import cv2

from arg_utils import check_file_existance
from image_utils import normalize_image

from logging import getLogger
logger = getLogger(__name__)

def calc_adjust_fsize(f_height, f_width, height, width):
    """
    Calculate the adjusted frame size to maintain aspect ratio.

    Parameters
    ----------
    f_height : int
        Original frame height.
    f_width : int
        Original frame width.
    height : int
        Target height.
    width : int
        Target width.

    Returns
    -------
    int, int
        New height and width that maintain aspect ratio.
    """
    scale = np.max((f_height / height, f_width / width))
    return int(scale * height), int(scale * width)


def adjust_frame_size(frame, height, width):
    """
    Adjust the frame size by adding padding to fit the target dimensions.

    Parameters
    ----------
    frame : numpy array
        The input frame.
    height : int
        Target height.
    width : int
        Target width.

    Returns
    -------
    img : numpy array
        Frame with adjusted proportions.
    resized_img : numpy array
        Resized version of the adjusted frame.
    """
    f_height, f_width = frame.shape[0], frame.shape[1]
    scale = np.max((f_height / height, f_width / width))

    # Create a blank image with target dimensions
    img = np.zeros((int(round(scale * height)), int(round(scale * width)), 3), np.uint8)
    start = (np.array(img.shape) - np.array(frame.shape)) // 2

    # Place the frame in the center of the blank image
    img[start[0]: start[0] + f_height, start[1]: start[1] + f_width] = frame
    resized_img = cv2.resize(img, (width, height))
    return img, resized_img


def cut_max_square(frame: np.array) -> np.array:
    """
    Crop the maximum square area from the center of the frame.

    Parameters
    ----------
    frame : numpy array
        Input frame.

    Returns
    -------
    frame_square : numpy array
        Cropped square area from the frame.
    """
    frame_height, frame_width, _ = frame.shape
    frame_size_min = min(frame_width, frame_height)

    # Calculate cropping coordinates
    if frame_width >= frame_height:
        x, y = frame_width // 2 - frame_height // 2, 0
    else:
        x, y = 0, frame_height // 2 - frame_width // 2

    # Crop and return the center square
    frame_square = frame[y: (y + frame_size_min), x: (x + frame_size_min)]
    return frame_square


def preprocess_frame(frame, input_height, input_width, data_rgb=True, normalize_type='255'):
    """
    Pre-process the frame for model input by resizing, normalizing, and adjusting channels.

    Parameters
    ----------
    frame : numpy array
        Input frame.
    input_height : int
        Target height.
    input_width : int
        Target width.
    data_rgb : bool, default=True
        Convert frame to RGB if True, grayscale if False.
    normalize_type : str, default='255'
        Type of normalization to apply ('255', '127.5', 'ImageNet', 'None').

    Returns
    -------
    img : numpy array
        Adjusted frame.
    data : numpy array
        Preprocessed frame ready for model input.
    """
    img, resized_img = adjust_frame_size(frame, input_height, input_width)

    # Convert frame to RGB if needed
    if data_rgb:
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

    # Normalize the frame
    data = normalize_image(resized_img, normalize_type)

    # Adjust channels for RGB or grayscale input
    if data_rgb:
        data = np.rollaxis(data, 2, 0)
        data = np.expand_dims(data, axis=0).astype(np.float32)
    else:
        data = cv2.cvtColor(data.astype(np.float32), cv2.COLOR_BGR2GRAY)
        data = data[np.newaxis, np.newaxis, :, :]
    return img, data


def get_writer(savepath, height, width, fps=20, rgb=True):
    """
    Initialize a video writer for saving video or streaming.

    Parameters
    ----------
    savepath : str
        Path or network address to save or stream video.
    height : int
        Video frame height.
    width : int
        Video frame width.
    fps : int
        Frames per second.
    rgb : bool, default=True
        Set to True for color video, False for grayscale.

    Returns
    -------
    writer : cv2.VideoWriter
        Video writer object.
    """
    # Check if saving to network stream
    if re.match(r'localhost\:', savepath) or re.match(r'[0-9]+(?:\.[0-9]+){3}\:', savepath):
        bitrate = 10000000
        tcp = True
        ip, port = savepath.split(":")
        logger.info("Opening gstreamer with IP " + ip + " and port " + port)

        # Define encoder and sink for network streaming
        encoder = 'nvvidconv ! nvv4l2h264enc bitrate=' + str(bitrate) + ' insert-sps-pps=true maxperf-enable=1'
        sink = ('appsrc ! video/x-raw,format=BGR ! queue ! videoconvert ! video/x-raw,format=BGRx ! ' + encoder +
                ' ! rtph264pay config-interval=1 ! ' +
                ('gdppay ! tcpserversink' if tcp else 'udpsink') + ' host=' + ip + ' port=' + port)
        
        writer = cv2.VideoWriter(sink, 0, int(fps), (width, height))
        if not writer.isOpened():
            logger.error("Unable to open gstreamer stream.")
            sys.exit(0)
        return writer

    # Save to file if not streaming
    if os.path.isdir(savepath):
        savepath = savepath + "/out.mp4"

    writer = cv2.VideoWriter(
        savepath,
        cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),  # MP4 format
        fps,
        (width, height),
        isColor=rgb
    )
    return writer


class BaslerCameraCapture:
    """
    Class for capturing video frames from a Basler camera using pypylon.
    """
    def __init__(self):
        self.camera = None
        self.converter = None

    def start_capture(self):
        """
        Initialize and start capturing from the Basler camera.
        """
        from pypylon import pylon
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.camera.Open()
        
        # Set capture parameters
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    def read(self):
        """
        Capture a frame from the Basler camera.

        Returns
        -------
        bool, numpy array
            Success flag and captured frame if successful.
        """
        from pypylon import pylon
        if self.camera is None:
            raise Exception("Capture not started")

        grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grab_result.GrabSucceeded():
            converted_frame = self.converter.Convert(grab_result)
            rgb_frame = converted_frame.GetArray()
            grab_result.Release()
            return True, rgb_frame
        else:
            return False, None

    def stop_capture(self):
        """
        Stop and release the camera capture.
        """
        if self.camera is not None:
            self.camera.Close()
            self.camera = None


def get_capture(video):
    """
    Initialize video capture from a file path, webcam ID, or network stream.

    Parameters
    ----------
    video : str
        Video source identifier (webcam ID or file path).

    Returns
    -------
    capture : cv2.VideoCapture or BaslerCameraCapture
        Video capture object.
    """
    try:
        video_id = int(video)
        # Open webcam by ID
        capture = cv2.VideoCapture(video_id)
        if not capture.isOpened():
            logger.error(f"Webcam with ID {video_id} not found")
            sys.exit(0)

    except ValueError:
        # Open video file or network stream if path is provided
        if "rtsp://" in video:
            capture = cv2.VideoCapture(video)
        elif "basler" in video:
            capture = BaslerCameraCapture()
            capture.start_capture()
        elif check_file_existance(video):
            capture = cv2.VideoCapture(video)

    return capture
