# 🧠 Valorant AI Object Detection
An AI-powered object detection system designed for real-time performance in Valorant using YOLOv8.

## 📂 Project Structure
```text
├── video_inference.py             # Run detection on uploaded video
├── video_inference_output.py      # Same as above, but saves annotated video
├── realtime_inference.py          # Live detection on screen (centered region)
├── data.yaml                      # YOLOv8 dataset config
├── README.md                      # This file
├── /runs                          # YOLO training results (model weights)
└── /videos                        # Your input videos
``` 

## 🛠️ Technologies
Python 3.8+

YOLOv8 (Ultralytics)

OpenCV

PyAutoGUI

MSS (for screen capture, optional)

PyQt5 (for overlay)

Win32 API (optional overlay tricks)

## ⚙️ Setup
Install dependencies
```text
pip install ultralytics opencv-python pyautogui pygetwindow pyqt5 mss
``` 

Make sure your model path is set correctly
In your scripts:
```text
model = YOLO("C:/Users/Admin/ultralytics/runs/detect/train/weights/best.pt")
```

## 🧪 Training Instructions
Prepare your dataset
Make sure your dataset is in the following structure:
```text
/dataset
├── images
│   ├── train
│   └── val
├── labels
│   ├── train
│   └── val
└── data.yaml
```
Check your data.yaml file
```text
train: dataset/images/train
val: dataset/images/val
nc: 1  # number of classes
names: ['enemy']  # replace with your class names
```
Train with YOLOv8
```text
pip install ultralytics
yolo task=detect mode=train model=yolov8n.pt data=dataset/data.yaml epochs=50 imgsz=640
```
After training completes, your best model will be saved to:
```text
runs/detect/train/weights/best.pt
```

## 🎮 Modes of Operation

### 🎥 Inference on Video
```text
python video_inference_output.py --video path/to/your_clip.mp4
```
→ Saves output as your_clip_output.mp4

### 🔴 Real-Time Detection
```text
python realtime_inference.py
```
→ Uses screen capture on the center of your monitor

## 🚨 Disclaimer
This tool is strictly for educational and offline use.
Using real-time object detection tools in online games may violate Terms of Service and result in bans.

