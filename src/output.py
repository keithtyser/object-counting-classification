# src/output.py
import cv2

def draw_output(image, classified_objects, counts):
    """
    Draw bounding boxes and labels on the output image and display counts.

    Parameters:
    - image: Input image.
    - classified_objects: List of classified objects.
    - counts: Dictionary with counts of each object type.

    Returns:
    - image: Image with bounding boxes and labels drawn.
    """
    for obj in classified_objects:
        label, confidence, box = obj
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f'{label} ({confidence:.2f})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    y_offset = 30
    for label, count in counts.items():
        cv2.putText(image, f'{label}: {count}', (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        y_offset += 30
    
    return image

def generate_report(all_counts):
    """
    Generate and print a summary report of the counts.

    Parameters:
    - all_counts: Dictionary with image file names as keys and counts as values.
    """
    total_counts = {'person': 0, 'bicycle': 0, 'car': 0}

    for counts in all_counts.values():
        for key, value in counts.items():
            total_counts[key] += value

    print("Summary Report:")
    for label, count in total_counts.items():
        print(f'{label}: {count}')
