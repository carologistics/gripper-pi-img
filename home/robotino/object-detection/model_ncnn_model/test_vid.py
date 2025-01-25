import cv2
import ncnn
from utils import ncnn_inference, DEFAULT_CLASS_NAMES
import numpy as np

# Confidence threshold
threshold = 0.2
# Non-max suppression threshold
nms_threshold = 0.7
# Set the input image size
target_size = 640

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Initialize NCNN model
net = ncnn.Net()
net.load_param(r"model.ncnn.param")
net.load_model(r"model.ncnn.bin")


def draw_detections(frame, detections):
    for detection in detections:
        box = detection.bounds
        print(type(box))
        cv2.rectangle(frame, (int(box.x), int(box.y)), 
                      (int(box.x + box.width), int(box.y + box.height)), (0, 0, 255), 3)
        label = f"{detection.class_name} ({round(detection.confidence * 100, 2)}%)"
        cv2.putText(frame, label, (int(box.x), int(box.y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # infer
    detections, infer_time = ncnn_inference(frame_rgb, net, target_size, threshold, nms_threshold)
    if detections:
        print(f"Detection:{detections} Inference_time:{infer_time}")
          
    draw_detections(frame, detections)
    cv2.imshow("Frame with Detections", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()