import json
import os

def load_coco_annotations(annotation_file):
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    return annotations

def get_ground_truth_counts(annotations, image_ids):
    """
    Get ground truth counts for the specified image IDs.
    
    Parameters:
    - annotations: COCO annotations.
    - image_ids: List of image IDs to get ground truth counts for.
    
    Returns:
    - ground_truth_counts: Dictionary with image IDs as keys and object counts as values.
    """
    ground_truth_counts = {}
    for image_id in image_ids:
        image_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] == image_id]
        counts = {'person': 0, 'bicycle': 0, 'car': 0}
        for ann in image_annotations:
            category_id = ann['category_id']
            category_name = next(cat['name'] for cat in annotations['categories'] if cat['id'] == category_id)
            if category_name in counts:
                counts[category_name] += 1
        ground_truth_counts[image_id] = counts
    return ground_truth_counts

def main():
    # Correct the path to the annotations file
    annotation_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'annotations', 'instances_val2017.json'))
    
    # Load COCO annotations
    annotations = load_coco_annotations(annotation_file)
    
    # List of image IDs (you can get these from the file names or another source)
    image_ids = [139, 285, 632]  # Example image IDs
    
    # Get ground truth counts for the specified images
    ground_truth_counts = get_ground_truth_counts(annotations, image_ids)
    
    # Print ground truth counts
    print(json.dumps(ground_truth_counts, indent=4))

if __name__ == "__main__":
    main()
