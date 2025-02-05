import torch
from PIL import Image

# Load YOLOv5 model (pretrained on COCO dataset)111
model = torch.hub.load('ultralytics/yolov5', 'yolov5m')


# Load a local example image
image_path = r"C:\project\INPUT\maadu.jpg"  # Use raw string (r"")
img = Image.open(image_path)
results = model(img)

# Print the count of detected objects
print("Detected cattle count:", len(results.xyxy[0]))  # results.xyxy[0] contains the bounding boxes

# Save the image with detections as a PNG file
save_path = 'C:\project\yolov5\runs\detect\exp'  
results.save(save_path)
