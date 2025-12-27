# Remove-Background
🖼️ High-Performance AI Background Removal Tool
This project showcases a professional, optimized solution for foreground-background segmentation in both static images and video streams. Developed as a Master's research project, the core implementation utilizes the state-of-the-art DIS (or IS-Net) segmentation model to achieve highly accurate object isolation.
The entire inference pipeline is meticulously optimized with ONNX Runtime for high-speed, efficient performance, making it suitable for real-time applications and batch processing workflows.


![Image Alt](https://github.com/ipaaco/Remove-Background/blob/237b4d75c249d41f61cdc132647714db820f9b9b/Screenshot%202025-12-27%20035619.png)

✨ Key Capabilities
Model: DIS/IS-Net architecture for robust segmentation.

Performance: Optimized inference via ONNX Runtime and built-in benchmark capabilities.

Versatility: Seamlessly handles image files, batch image lists, and live video/webcam input.

Output: Generates four-channel (BGRA) PNG images with a transparent background, ready for compositing.

🛠️ Setup and Installation
Prerequisites
Python 3.x

The required Python libraries.

Pre-trained ONNX model files.

1. Environment Setup
It is highly recommended to use a virtual environment:

pip install onnxruntime opencv-python numpy

2. Acquiring Model Weights
The required ONNX model files (isnet-general-use.onnx and dis.onnx) are hosted externally. Please download the weights from the following link and place them directly in your project's root directory:

Model Weights Download Link: https://mega.nz/folder/1T50TIpY#JS8cMj1lWx0aOE-89DqR9Q

3. Utility Modules
The main script relies on several custom utility modules. Ensure your project structure allows the script to access the following files within the remove_background_company/utils path:

webcamera_utils.py

image_utils.py (containing normalize_image)

detector_utils.py (containing load_image)

arg_utils.py

🚀 Usage
Image Mode
Runs segmentation on one or more static images, saving the result as a transparent PNG.
python main_script.py -i input/chair.jpg -s output/
Video Mode
To process a video file or live webcam input:

python main_script.py --video 0


python main_script.py --video input.mp4 -s output/output_video.mp4

💻 Technical Summary
Processing Pipeline
The system follows a three-stage pipeline to achieve background removal: Preprocessing, Inference, and Post-processing.

Preprocessing: The input image is first resized to the model's expected square dimensions (default 1024 by 1024 pixels, controlled by --img-size). The image colors are then normalized by scaling the pixel values to prepare them for the neural network. Finally, the image data is rearranged into the specific Channel-Height-Width-Batch format that the AI model requires.

Inference: The prepared data is passed directly through the ONNX Runtime session. This executes the deep learning model (DIS/IS-Net), which generates a preliminary, lower-resolution segmentation mask.

Post-processing: The raw segmentation mask is resized back to the exact dimensions of the original input image. It is then refined using min-max normalization to ensure all mask values are clean and fall uniformly between 0 (fully transparent) and 1 (fully opaque).

Output Generation
The final segmented image is created by combining the refined mask with the original image data.

The normalized mask is essentially multiplied pixel-by-pixel with the original color image (BGR channels) to keep only the foreground object. For static images, the result is saved as a four-channel image (BGRA): the three color channels contain the isolated object, and the fourth channel (the Alpha Channel) contains the mask itself, resulting in a perfectly transparent background.
