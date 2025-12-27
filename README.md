# Remove-Background
High-Performance AI Background Removal Tool
This repository showcases a professional, optimized solution for foreground-background segmentation in both static images and video streams. Developed as a Master's research project, the core implementation utilizes the state-of-the-art DIS (or IS-Net) segmentation model to achieve highly accurate object isolation.

The entire inference pipeline is meticulously optimized with ONNX Runtime for high-speed, efficient performance, making it suitable for real-time applications and batch processing workflows.

Key Capabilities:

Model: DIS/IS-Net architecture for robust segmentation.

Performance: Optimized inference via ONNX Runtime and built-in benchmark capabilities.

Versatility: Seamlessly handles image files, batch image lists, and live video/webcam input.

Output: Generates four-channel (BGRA) PNG images with a transparent background, ready for compositing.
