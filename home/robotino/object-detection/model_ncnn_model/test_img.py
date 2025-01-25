import ncnn
from utils import ncnn_inference
import numpy as np
from PIL import Image, ImageDraw

# Confidence threshold
threshold = 0.2
# Non-max suppression threshold
nms_threshold = 0.7
# Set the input image size
target_size = 640

imagepath = r"test.jpg"

net = ncnn.Net()

# Load NCNN model
net.load_param(r"model.ncnn.param")
net.load_model(r"model.ncnn.bin")

# Load the image as a numpy array using PIL
img = Image.open(imagepath)
frame_rgb = np.array(img)  # Convert the PIL image to a numpy array

# Option 1: Use default class names
outputs, inference_time = ncnn_inference(frame_rgb, net, target_size, threshold, nms_threshold)

# Option 2: Use custom class names
# custom_class_names = ["background", "object1", "object2", "object3"]
# outputs_custom = ncnn_inference(frame_rgb, net, target_size, threshold, nms_threshold, custom_class_names)

# Draw and display the detections (using either default or custom class names)
print(inference_time)
for output in outputs:
    print(f"{output}")


def draw_detections(image_path, detections):
    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)
        for detection in detections:
            box = detection.bounds
            draw.rectangle([box.x, box.y, box.x + box.width, box.y + box.height], outline="red", width=3)
            label = f"{detection.class_name} ({round(detection.confidence * 100, 2)}%)"
            draw.text((box.x, box.y), label, fill="red")
        img.show()

draw_detections(imagepath, outputs)  # Draw using default class names
# draw_detections(imagepath, outputs_custom)  # Draw using custom class names