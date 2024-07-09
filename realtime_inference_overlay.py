import cv2
import numpy as np
import pyautogui
from ultralytics import YOLO
from PIL import Image
import pygetwindow as gw
import mss
import win32gui
import win32ui
import win32con
import win32api

# Load the trained model
model = YOLO('C:/Users/Admin/ultralytics/runs/detect/train8/weights/best.pt')

# Set the title of your game window
game_window_title = 'VALORANT'
game_window = gw.getWindowsWithTitle(game_window_title)[0]
left, top, width, height = game_window.left, game_window.top, game_window.width, game_window.height

# Function to draw text and rectangles directly on the screen
def draw_overlay(hdc, text, x, y, rect, color=win32api.RGB(255, 0, 0)):
    font = win32ui.CreateFont({
        "name": "Arial",
        "height": 20,
        "weight": win32con.FW_BOLD
    })
    hdc.SelectObject(font)
    hdc.SetTextColor(color)
    hdc.TextOut(x, y, text)
    hdc.Rectangle(rect)

# Open a named window for display
cv2.namedWindow("YOLO Inference", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("YOLO Inference", cv2.WND_PROP_TOPMOST, 1)

with mss.mss() as sct:
    while True:
        # Capture the screen
        monitor = {"top": top, "left": left, "width": width, "height": height}
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Convert frame to PIL image
        image = Image.fromarray(frame)

        # Perform inference
        results = model.predict(source=image, imgsz=640)

        # Draw bounding boxes on the frame
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{box.conf[0]:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Overlay the frame on the game window
        hwnd = game_window._hWnd
        window_dc = win32gui.GetWindowDC(hwnd)
        dc_obj = win32ui.CreateDCFromHandle(window_dc)
        compatible_dc = dc_obj.CreateCompatibleDC()

        data_bitmap = win32ui.CreateBitmap()
        data_bitmap.CreateCompatibleBitmap(dc_obj, width, height)
        compatible_dc.SelectObject(data_bitmap)

        compatible_dc.BitBlt((0, 0), (width, height), dc_obj, (0, 0), win32con.SRCCOPY)

        # Draw overlay for each detection
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            draw_overlay(compatible_dc, label, x1, y1 - 20, (x1, y1, x2, y2))

        dc_obj.BitBlt((0, 0), (width, height), compatible_dc, (0, 0), win32con.SRCCOPY)
        win32gui.ReleaseDC(hwnd, window_dc)
        compatible_dc.DeleteDC()
        win32gui.DeleteObject(data_bitmap.GetHandle())
        dc_obj.DeleteDC()

        # Display the resulting frame (optional)
        cv2.imshow('YOLO Inference', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Close all OpenCV windows
cv2.destroyAllWindows()
