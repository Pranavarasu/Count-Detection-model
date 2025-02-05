import torch
import cv2
import numpy as np
import time

# Load YOLOv5 model (pretrained on COCO dataset)
print("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
print("Model loaded")

# Initialize video capture using different backend
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use OpenCV DirectShow backend

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Add delay before starting the capture
time.sleep(1)
print("Starting live detection...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb)

    for i, (xmin, ymin, xmax, ymax, conf, class_id) in enumerate(results.xyxy[0]):
        class_name = model.names[int(class_id)]
        cv2.putText(frame, f'{class_name}', (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

    cv2.imshow('Live Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
