import cv2
import numpy as np
import pyautogui
from ultralytics import YOLO
from PIL import Image
import time
import torch

# Load the trained model
model = YOLO('C:/Users/Admin/ultralytics/runs/detect/train8/weights/best.pt')

# Define the region of the screen to capture (x, y, width, height)
screen_width, screen_height = 3440, 1440
dpi = 96  # Adjust according to your screen's DPI if needed
capture_size = int(12 * dpi)  # 2 inches in pixels
center_x, center_y = screen_width // 2, screen_height // 2
region = (center_x - capture_size // 2, center_y - capture_size // 2, capture_size, capture_size)

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Target frame rate
target_fps = 60
frame_time = 1.0 / target_fps  # Time per frame in seconds

# Open a named window for display
cv2.namedWindow("YOLO Inference", cv2.WINDOW_NORMAL)

while True:
    start_time = time.time()

    # Capture the screen
    screenshot = pyautogui.screenshot(region=region)
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert frame to PIL image
    image = Image.fromarray(frame)

    # Perform inference
    results = model.predict(source=image, imgsz=640, device=device)

    # Draw bounding boxes on the frame
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{box.conf[0]:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('YOLO Inference', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Calculate the time taken for this frame
    elapsed_time = time.time() - start_time

    # Sleep for the remaining time to achieve the target frame rate
    time_to_sleep = frame_time - elapsed_time
    if time_to_sleep > 0:
        time.sleep(time_to_sleep)

# Close all OpenCV windows
cv2.destroyAllWindows()
