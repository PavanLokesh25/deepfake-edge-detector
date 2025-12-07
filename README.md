# YOLOv8-Face + EfficientNet-B0 | Jetson Nano / Jetson Orin Compatible

This repository contains an offline, real-time deepfake detection system designed to run efficiently on edge devices such as NVIDIA Jetson Nano and Jetson Orin. The system combines:

* **YOLOv8-Face** → fast and accurate face detection
* **EfficientNet-B0** → lightweight deepfake classifier
* **Temporal smoothing + face tracking** → stable prediction over time
* **ONNX + TensorRT (Jetson)** → optimized runtime

This project eliminates dependence on cloud servers, ensuring privacy, low latency, and real-time deepfake detection.

## Features

* Offline deepfake detection
* Works on CPU, GPU, Jetson Nano, Jetson Orin
* YOLOv8-Face for fast face detection
* EfficientNet-B0 deepfake classifier
* Exponential smoothing for stable predictions
* Face tracking to maintain identity across frames
* ONNX export for TensorRT acceleration
* Real-time visualization with OpenCV

## Repository Structure
```
├── models/
│   ├── yolov8n-face.onnx
│   ├── efficientnet_model.onnx
│   ├── efficientnet_model.pt
│
├── scripts/
│   ├── deepfake_detector_colab.py
│   ├── deepfake_runtime_jetson.py
│
├── sample_videos/
│   ├── demo.mp4
│
├── README.md




> **Note:** For Jetson devices, dependencies are installed differently (see Jetson setup section below).

## Models

### YOLOv8-Face

* Detects faces in each frame
* Exported to ONNX for Jetson

Download or use the included file:
```
models/yolov8n-face.onnx
```

### EfficientNet-B0 Deepfake Classifier

* Trained using real vs fake face crops
* Outputs probability: fake vs real
```
models/efficientnet_model.pt
models/efficientnet_model.onnx
```

## How the Detection Pipeline Works

1. **Frame Sampling** - Process every Nth frame for speed
2. **Face Detection (YOLOv8-Face)** - Outputs bounding boxes + scores
3. **Face Tracking** - Ensures the same face keeps the same ID across frames
4. **Classification (EfficientNet-B0)** - Each face crop → fake probability
5. **Temporal Smoothing** - Exponential smoothing stabilizes predictions:
```
   smooth = prev * (1 - α) + current * α
```
6. **Final Decision** - If final probability > 0.5 → deepfake detected

## Run Deepfake Detection (PC)
```bash
python scripts/deepfake_detector_colab.py --video path/to/video.mp4
```

## Running on Jetson Nano / Jetson Orin

### Transfer ONNX Models

Copy these files to your Jetson:
```
models/yolov8n-face.onnx
models/efficientnet_model.onnx
scripts/deepfake_runtime_jetson.py
```

Use SCP:
```bash
scp models/*.onnx jetson@<ip-address>:/home/jetson/
scp scripts/deepfake_runtime_jetson.py jetson@<ip-address>:/home/jetson/
```

### Install Jetson Dependencies
```bash
sudo apt update
sudo apt install python3-pip python3-opencv libopencv-dev
pip3 install numpy
pip3 install onnx onnxruntime
pip3 install ultralytics --no-dependencies
```

### Run the Jetson Optimized Script
```bash
python3 deepfake_runtime_jetson.py --video input.mp4
```

This will:
* Load ONNX models
* Detect faces
* Classify them


