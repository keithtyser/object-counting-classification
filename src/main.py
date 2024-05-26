import cv2
import os
import json
from tqdm import tqdm
from preprocess import preprocess_image
from detect import load_yolo_model, detect_objects
from classify import classify_objects
from count import count_objects
from output import draw_output, generate_report

def main():
    # Define the base path to the test images directory
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'test_images', 'val2017'))
    
    # Gather all image files in the directory
    input_images = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Ensure there are images to process
    if not input_images:
        print("No images found in the directory.")
        return

    all_counts = {}

    # Load YOLO model and classes
    net, classes = load_yolo_model()

    # Process each image with a progress bar
    for img_file in tqdm(input_images, desc="Processing images"):
        # Load and preprocess image
        image = cv2.imread(img_file)
        if image is None:
            print(f"Failed to load image: {img_file}")
            continue
        preprocessed_image = preprocess_image(image)

        # Detect objects
        detections = detect_objects(preprocessed_image, net, classes)

        # Classify objects
        classified_objects = classify_objects(detections, classes)

        # Count objects
        counts = count_objects(classified_objects)
        all_counts[os.path.basename(img_file)] = counts

        # Draw output and save image
        output_image = draw_output(image, classified_objects, counts)
        output_base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'output_images'))
        os.makedirs(output_base_path, exist_ok=True)
        cv2.imwrite(os.path.join(output_base_path, f'output_{os.path.basename(img_file)}'), output_image)

    # Save predicted counts to a JSON file
    predicted_counts_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'predicted_counts.json'))
    with open(predicted_counts_file, 'w') as f:
        json.dump(all_counts, f)

    # Generate and print summary report
    generate_report(all_counts)

if __name__ == "__main__":
    main()
