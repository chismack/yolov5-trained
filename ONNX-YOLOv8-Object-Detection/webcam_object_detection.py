import cv2
import numpy as np
import os
from datetime import datetime
from yolov8 import YOLOv8

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize YOLOv8 object detector
model_path = r"D:\\yolov8\\custom320.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

capture_dir = "captured_images"
os.makedirs(capture_dir, exist_ok = True)

# List to store the points
points = []

def draw_polygon(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Detected Objects", draw_polygon)

while cap.isOpened():
    # Read frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Update object localizer
    boxes, scores, class_ids = yolov8_detector(frame)

    # Draw detections on the frame
    combined_img = yolov8_detector.draw_detections(frame)

    # Draw the polygon if 4 points are recorded
    if len(points) == 4:
        overlay = combined_img.copy()
        points_np = np.array(points, np.int32)
        points_np = points_np.reshape((-1, 1, 2))

        for box in boxes:
            x, y, w, h = box
            box_center = (x + w / 2, y + h / 2)

            # Check if the center of the bounding box is inside the polygon
            if cv2.pointPolygonTest(points_np, box_center, False) >= 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = os.path.join(capture_dir, f"captured_image_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                print(f"Image captured: {filename}")
                image_captured = True
                capture_time = datetime.now()
                break

        cv2.polylines(combined_img, [points_np], isClosed=True, color=(0, 255, 0), thickness=2)

    cv2.imshow("Detected Objects", combined_img)

    # Press key q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
