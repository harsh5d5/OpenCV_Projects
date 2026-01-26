# OpenCV Projects

A collection of Computer Vision projects implementing various features like hand tracking, gesture-based control, and object movement.

## 🚀 About Computer Vision

Computer Vision (CV) is a field of artificial intelligence that enables computers and systems to derive meaningful information from digital images, videos, and other visual inputs.

### 🖼️ OpenCV (Open Source Computer Vision Library)
OpenCV is the most popular library for real-time computer vision. It contains thousands of optimized algorithms for:
- **Image Processing**: Filtering, transformations, and color space conversions.
- **Feature Detection**: Detection of corners, blobs, and lines.
- **Object Detection**: Detecting pre-defined shapes or objects.
- **Video Analysis**: Motion tracking and background subtraction.

### 🐍 Pillow (PIL Fork)
Pillow is a powerful library for opening, manipulating, and saving many different image file formats. While OpenCV is better for real-time processing, Pillow is excellent for:
- Simple batch processing.
- Image archiving.
- Drawing text and Basic shapes on images.
- Basic transformations like resizing and rotating.

### 🤖 Object Detection Models: YOLO & DETR

#### YOLO (You Only Look Once)
YOLO is a state-of-the-art, real-time object detection system. It is incredibly fast and accurate because it treats detection as a single regression problem, straight from image pixels to bounding box coordinates and class probabilities.

#### DETR (DEtection TRansformer)
DETR is a transformer-based approach to object detection. It simplifies the detection pipeline by removing the need for many hand-designed components like non-maximum suppression or anchor generation.

---

## 📂 Included Projects

### 1. Hand Box Moving (`Hand_box_moveing`)
An interactive project where users can move a virtual box on the screen using hand gestures detected via a webcam.

### 2. PPT Controller (`ppt_controller`)
Easily navigate through PowerPoint presentations using hand gestures. No need for a remote or mouse!

### 3. Virtual Mouse (`virtual_mouse`)
Control your computer's cursor and perform clicks using hand movements. A touchless way to interact with your machine.

---

## 🛠️ Requirements

- Python 3.x
- OpenCV (`opencv-python`)
- MediaPipe
- PyAutoGUI (for mouse control)
- Pillow

## 📜 License
This repository is open-source. Feel free to use and contribute!
