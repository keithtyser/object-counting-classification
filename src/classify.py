# src/classify.py

def classify_objects(detections, classes):
    """
    Classify detected objects into predefined categories.

    Parameters:
    - detections: List of detected objects from YOLO.
    - classes: List of class names from COCO.

    Returns:
    - classified_objects: List of classified objects with labels, confidence, and bounding box.
    """
    predefined_categories = {'person', 'bicycle', 'car'}
    classified_objects = []

    for detection in detections:
        label, confidence, box = detection
        if label in predefined_categories:
            classified_objects.append((label, confidence, box))
    
    return classified_objects
