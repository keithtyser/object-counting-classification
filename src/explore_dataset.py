# explore_dataset.py
import os
import json

# Path to COCO annotations
annotations_file = r'C:\Users\Keith\Documents\Code\object-counting-classification\data\annotations\instances_val2017.json'

# Load annotations
with open(annotations_file, 'r') as f:
    annotations = json.load(f)

# Get category names
categories = annotations['categories']
category_names = [category['name'] for category in categories]

# Print category names
print("Categories in COCO dataset:")
print(category_names)

# Check if the required categories are present
required_categories = ['car', 'bicycle', 'person']
for category in required_categories:
    if category in category_names:
        print(f"{category} is present in the dataset.")
    else:
        print(f"{category} is NOT present in the dataset.")
