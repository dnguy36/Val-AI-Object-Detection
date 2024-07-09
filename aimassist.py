import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import mss
import serial

arduino = serial.Serial(port='COM3', baudrate=9600, timeout=.1)

model = YOLO('C:/Users/Admin/ultralytics/runs/detect/train8/weights/best.pt')

monitor = {"top": 100, "left": 100, "width": 800, "height": 600}

# Define 1x1 inch area in pixels
inch_size = 96
crosshair_area = {
    "left": monitor["width"] // 2 - inch_size // 2,
    "top": monitor["height"] // 2 - inch_size // 2,
    "right": monitor["width"] // 2 + inch_size // 2,
    "bottom": monitor["height"] // 2 + inch_size // 2
}

def send_to_arduino(data):
    arduino.write(bytes(data, 'utf-8'))
    arduino.flush()

cv2.namedWindow("YOLO Inference", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("YOLO Inference", cv2.WND_PROP_TOPMOST, 1)

with mss.mss() as sct:
    while True:
        # Capture screen
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        image = Image.fromarray(frame)

        # Perform inference
        results = model.predict(source=image, imgsz=640)

        # Check for detections within the 1x1 inch area around the crosshair
        detected = False
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            if (crosshair_area["left"] <= center_x <= crosshair_area["right"] and
                crosshair_area["top"] <= center_y <= crosshair_area["bottom"]):
                detected = True
                screen_center_x = monitor["width"] // 2
                screen_center_y = monitor["height"] // 2
                delta_x = center_x - screen_center_x
                delta_y = center_y - screen_center_y

                # Send adjustments to Arduino
                command = f'{delta_x},{delta_y}\n'
                send_to_arduino(command)

                # Draw bounding box and center point
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
                cv2.putText(frame, f'{delta_x},{delta_y}', (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                break

        cv2.rectangle(frame, (crosshair_area["left"], crosshair_area["top"]), (crosshair_area["right"], crosshair_area["bottom"]), (255, 0, 0), 2)

        cv2.imshow('YOLO Inference', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
