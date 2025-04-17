
import cv2
import torch
from ultralytics import YOLO
import argparse
import os

# Parse command-line argument for video path
parser = argparse.ArgumentParser(description="YOLOv8 Inference on Video File")
parser.add_argument("--video", type=str, required=True, help="Path to input video file")
args = parser.parse_args()

# Load the trained YOLO model
model = YOLO('C:/Users/Admin/ultralytics/runs/detect/train7/weights/best.pt')

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Open video file
cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Output video setup
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
filename = os.path.splitext(os.path.basename(args.video))[0]
out = cv2.VideoWriter(f"{filename}_output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

cv2.namedWindow("YOLO Video Inference", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)[0]
    annotated_frame = results.plot()

    # Show and save the annotated frame
    cv2.imshow("YOLO Video Inference", annotated_frame)
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
