from PIL import Image
import ncnn
import numpy as np
import time

# Default class names
DEFAULT_CLASS_NAMES = [".", "conveyor", "slide", "workpiece"]

class Rect:
    def __init__(self, x: float, y: float, width: float, height: float):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    @property
    def size(self):
        return Size(self.width, self.height)
    
    @property
    def is_empty(self):
        return self.width <= 0 or self.height <= 0
    
    @property
    def origin(self):
        return Point(self.x, self.y)
    
    @property
    def center(self):
        return Point(self.x + self.width / 2, self.y + self.height / 2)
    
    @property
    def area(self):
        return self.width * self.height
    
    def contains(self, point):
        return self.x <= point.x <= self.x + self.width and self.y <= point.y <= self.y + self.height
    
    def __str__(self) -> str:
        return f'{{x: {self.x}, y: {self.y}, w: {self.width}, h: {self.height}}}'
    
    def __repr__(self) -> str:
        return str(self)


class Size:
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height


class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class DetectedObject:
    @property
    def bounds(self):
        return Rect(round(self.xmin), round(self.ymin), round(self.xmax - self.xmin), round(self.ymax - self.ymin))

    def __init__(self, class_id: int, confidence: float, xmin: float, ymin: float, xmax: float, ymax: float, class_names=None):
        self.class_id = class_id
        self.confidence = confidence
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        # Use provided class names or the default
        self.class_names = class_names or DEFAULT_CLASS_NAMES

    @property
    def class_name(self):
        return self.class_names[self.class_id]
        
    def __str__(self) -> str:
        return f"Object(class={self.class_name}, conf={round(100*self.confidence,2)}, Box ={self.bounds})"

    def __repr__(self) -> str:
        return str(self)


def preprocess_image(image, target_size):
    img = image.convert("RGB")
    old_size = img.size
    ratio = float(target_size) / max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    img = img.resize(new_size, Image.BICUBIC)
    with Image.new("RGB", (target_size, target_size)) as new_img:
        new_img.paste(img, ((target_size-new_size[0])//2, (target_size-new_size[1])//2))
    return new_img


def nms(boxes, threshold):
    boxes.sort(key=lambda x: x.confidence, reverse=True)
    selected = []
    active = [True] * len(boxes)
    num_active = len(active)
    done = False
    i = 0
    while i < len(boxes) and not done:
        if active[i]:
            box_a = boxes[i]
            selected.append(box_a)
            if len(selected) >= 20:
                break
            for j in range(i+1, len(boxes)):
                if active[j]:
                    box_b = boxes[j]
                    if iou(box_a.bounds, box_b.bounds) > threshold:
                        active[j] = False
                        num_active -= 1
                        if num_active <= 0:
                            done = True
                            break
        i += 1
    return selected


def iou(a: Rect, b: Rect) -> float:
    area_a = a.area
    if area_a <= 0:
        return 0.0
    area_b = b.area
    if area_b <= 0:
        return 0.0
    intersection_min_x = max(a.x, b.x)
    intersection_min_y = max(a.y, b.y)
    intersection_max_x = min(a.x + a.width, b.x + b.width)
    intersection_max_y = min(a.y + a.height, b.y + b.height)
    intersection_area = max(intersection_max_x - intersection_min_x, 0) * max(intersection_max_y - intersection_min_y, 0)
    return intersection_area / (area_a + area_b - intersection_area)


def ncnn_inference(image_file, net, target_size, threshold, nms_threshold, class_names=None):
    # Convert the numpy array (image frame) to a PIL Image
    img = Image.fromarray(image_file)
    
    # We'll be letterboxing the image to fit the target size
    longer_side = max(img.size[0], img.size[1])
    scale = longer_side / target_size
    offset_x = (longer_side - img.size[0]) // 2
    offset_y = (longer_side - img.size[1]) // 2
    scaled_img = preprocess_image(img, target_size)
    
    image = ncnn.Mat.from_pixels(np.asarray(scaled_img), ncnn.Mat.PixelType.PIXEL_BGR, scaled_img.width, scaled_img.height)
    mean = [0, 0, 0]
    std = [1/255, 1/255, 1/255]
    image.substract_mean_normalize(mean=mean, norm=std)
    
    extractor = net.create_extractor()
    extractor.set_light_mode(True)
    extractor.input("in0", image)
    
    inference_start = time.perf_counter() #start time of ncnn inference
    out = ncnn.Mat()
    extractor.extract("out0", out)
    inference_end = time.perf_counter() #end time of ncnn inference
    out = np.asarray(out)

    detection_types = []
    for i in range(len(out[0])):
        cx, cy = out[0][i], out[1][i]
        w, h = out[2][i], out[3][i]
        scores = out[4:]
        class_id = np.argmax(scores[:, i])
        score = scores[class_id, i]
        
        if score > threshold:
            detection_types.append([score, cx - w/2, cy - h/2, cx + w/2, cy + h/2, w * h, class_id])

    extractor.clear()

    detection_types.sort(key=lambda x: x[0], reverse=True)
    detections = list(map(lambda x: DetectedObject(x[6], x[0], x[1] * scale - offset_x, x[2] * scale - offset_y, x[3] * scale - offset_x, x[4] * scale - offset_y, class_names), detection_types))
    
    detections = list(filter(lambda x: x.confidence >= threshold, nms(detections, nms_threshold)))
    
    ncnn_elapsed_time = (inference_end - inference_start) * 1000 
    return detections, ncnn_elapsed_time
