# src/detect.py
import cv2
import numpy as np
import os

def load_yolo_model():
    net = cv2.dnn.readNet(os.path.join('..', 'models', 'yolov3.weights'), os.path.join('..', 'models', 'yolov3.cfg'))
    with open(os.path.join('..', 'models', 'coco.names'), 'r') as f:
        classes = f.read().strip().split('\n')
    return net, classes

def detect_objects(image, net, classes):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    # Prepare the image for the network
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)
    
    height, width = image.shape[:2]
    boxes = []
    confidences = []
    class_ids = []
    
    for detection in detections:
        for object in detection:
            scores = object[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Filter out weak detections
                center_x = int(object[0] * width)
                center_y = int(object[1] * height)
                w = int(object[2] * width)
                h = int(object[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply non-maxima suppression to suppress weak overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            detections.append((classes[class_ids[i]], confidences[i], box))
    
    return detections
