# 🧠 Valorant AI Object Detection
An AI-powered object detection system designed for real-time performance in Valorant using YOLOv8.

## 📂 Project Structure
```text
├── video_inference_output.py      # Run detection on uploaded video and saves annotated video
├── realtime_inference.py          # Live detection on external screen (centered region)
├── realtime_inference_overlay.py  # Live detection overlay
├── aimassist.py                   # Calculates target offset and sends data to Arduino
├── arduino.cpp                    # Receives serial input and moves mouse via Arduino HID
├── data.yaml                      # YOLOv8 dataset config
├── README.md                      # This file
├── /runs                          # YOLO training results (model weights)
``` 

## 🛠️ Technologies
Python 3.8+

YOLOv8 (Ultralytics)

OpenCV

PyAutoGUI/MSS

PyQt5 (for overlay)

Arduino (Leonardo/Micro or other HID-compatible boards

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
### Capture your own dataset

Take screenshots or screen recordings of Valorant enemies

Extract frames or crop individual examples manually

### Label the images

Use LabelImg or Roboflow Annotate

Save labels in YOLO format (each .jpg or .png should have a corresponding .txt)

### Organize your dataset into folders
```text
/dataset
├── train
│   ├── image1.jpg
│   ├── image1.txt
│   └── ...
├── val
│   ├── image2.jpg
│   ├── image2.txt
│   └── ...
├── test         # (optional) run post-training predictions on this set
│   ├── image3.jpg
│   ├── image3.txt
│   └── ...
└── data.yamltext
```
Each image should have a corresponding .txt file with YOLO labels

You can optionally add a test folder for future evaluation (not used during training by default)

### Create your data.yaml file
```text
train: dataset/images/train
val: dataset/images/val
nc: 1  # number of classes
names: ['enemy']  # replace with your class names
```
### Train with YOLOv8
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

🎯 Aim Assist Mode
```text
python aimassist.py
```
→ Reads YOLO detections, calculates offset, and sends serial data to arduino.cpp

## 🚨 Disclaimer
This tool is strictly for educational and offline use.
Using real-time object detection tools in online games may violate Terms of Service and result in bans.

