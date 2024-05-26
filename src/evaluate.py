import json
import os

def load_coco_annotations(annotation_file):
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    return annotations

def get_ground_truth_counts(annotations, image_ids):
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

def calculate_metrics(predicted_counts, ground_truth_counts):
    tp = {'person': 0, 'bicycle': 0, 'car': 0}
    fp = {'person': 0, 'bicycle': 0, 'car': 0}
    fn = {'person': 0, 'bicycle': 0, 'car': 0}
    
    for img, gt_counts in ground_truth_counts.items():
        if img in predicted_counts:
            pred_counts = predicted_counts[img]
            for obj in tp.keys():
                tp[obj] += min(pred_counts.get(obj, 0), gt_counts.get(obj, 0))
                fp[obj] += max(0, pred_counts.get(obj, 0) - gt_counts.get(obj, 0))
                fn[obj] += max(0, gt_counts.get(obj, 0) - pred_counts.get(obj, 0))
        else:
            for obj in fn.keys():
                fn[obj] += gt_counts.get(obj, 0)
    
    precision = {obj: tp[obj] / (tp[obj] + fp[obj]) if (tp[obj] + fp[obj]) > 0 else 0 for obj in tp.keys()}
    recall = {obj: tp[obj] / (tp[obj] + fn[obj]) if (tp[obj] + fn[obj]) > 0 else 0 for obj in tp.keys()}
    f1_score = {obj: 2 * (precision[obj] * recall[obj]) / (precision[obj] + recall[obj]) if (precision[obj] + recall[obj]) > 0 else 0 for obj in tp.keys()}
    
    return precision, recall, f1_score

def main():
    # Correct the path to the annotations file
    annotation_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'annotations', 'instances_val2017.json'))
    
    # Load COCO annotations
    annotations = load_coco_annotations(annotation_file)
    
    # List of image IDs (you can get these from the file names or another source)
    image_ids = [int(f.split('.')[0]) for f in os.listdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'test_images', 'val2017'))) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Get ground truth counts for the specified images
    ground_truth_counts = get_ground_truth_counts(annotations, image_ids)
    
    # Load predicted counts from the main script output
    predicted_counts_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'predicted_counts.json'))
    with open(predicted_counts_file, 'r') as f:
        predicted_counts = json.load(f)
    
    # Convert filenames in predicted_counts to integers to match image_ids
    predicted_counts = {int(k.split('.')[0]): v for k, v in predicted_counts.items()}

    # Calculate metrics
    precision, recall, f1_score = calculate_metrics(predicted_counts, ground_truth_counts)
    
    # Print metrics
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1_score)

if __name__ == "__main__":
    main()
