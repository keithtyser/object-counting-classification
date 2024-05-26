# src/test_detection.py
import cv2
import matplotlib.pyplot as plt
from detect import load_yolo_model, detect_objects

def visualize_detection(image_path):
    """
    Visualize object detection on a sample image.
    
    Parameters:
    - image_path: Path to the input image.
    """
    # Load the YOLO model and class names
    net, classes = load_yolo_model()
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    # Detect objects
    detections = detect_objects(image, net, classes)
    
    # Draw bounding boxes
    for detection in detections:
        label, confidence, box = detection
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f'{label} ({confidence:.2f})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Convert image to RGB for displaying with matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Display the image
    plt.figure(figsize=(10, 5))
    plt.title('Object Detection')
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Paths to sample images
    sample_images = [
        '../data/test_images/val2017/000000000139.jpg',
        '../data/test_images/val2017/000000000285.jpg',
        '../data/test_images/val2017/000000000632.jpg'
    ]
    
    # Visualize detection on sample images
    for image_path in sample_images:
        visualize_detection(image_path)
