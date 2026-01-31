# Age and Gender Prediction

This project uses OpenCV's DNN module and pre-trained Caffe models to predict the age and gender of people detected via a webcam.

## 🚀 Features
- Real-time face detection.
- Gender classification (Male/Female).
- Age estimation (categorized into ranges).
- Visual feedback with bounding boxes and labels.

## 🛠️ Requirements
- `opencv-python`
- `requests` (for downloading models)

## 📂 Models Used
The project uses the following Caffe models:
- **Gender**: `gender_deploy.prototxt` and `gender_net.caffemodel`
- **Age**: `age_deploy.prototxt` and `age_net.caffemodel`
- **Face**: OpenCV's DNN Face Detector or Haar Cascades.

## 🏃 How to Run
1. Run `python download_models.py` to download the required models.
2. Run `python main.py` to start the prediction.
