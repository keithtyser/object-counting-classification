# src/test_preprocessing.py
import cv2
import os
from preprocess import preprocess_image
import matplotlib.pyplot as plt

def visualize_preprocessing(image_path):
    """
    Visualize the pre-processing steps on a sample image.
    
    Parameters:
    - image_path: Path to the input image.
    """
    # Load the original image
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    # Pre-process the image
    preprocessed_image = preprocess_image(original_image)
    
    # Convert images to RGB for displaying with matplotlib
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Display the original and pre-processed images
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(original_image_rgb)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title('Pre-processed Image')
    plt.imshow(preprocessed_image, cmap='gray')
    plt.axis('off')
    
    plt.show()

if __name__ == "__main__":
    # Paths to sample images
    sample_images = [
        '../data/test_images/val2017/000000000139.jpg',
        '../data/test_images/val2017/000000000285.jpg',
        '../data/test_images/val2017/000000000632.jpg'
    ]
    
    # Visualize pre-processing on sample images
    for image_path in sample_images:
        visualize_preprocessing(image_path)
