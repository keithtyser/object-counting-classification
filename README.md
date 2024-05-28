# Automated Object Counting and Classification System

## Overview
This project is a Python-based application that uses computer vision techniques to count and classify different objects from a set of images. The system differentiates between three types of objects: persons, bicycles, and cars. It processes images, detects and classifies objects, counts them, and generates output images with bounding boxes, labels, and counts.

## Features
- **Image Pre-processing**: Noise reduction and contrast enhancement.
- **Object Detection**: Uses YOLO (You Only Look Once) for detecting objects.
- **Object Classification**: Classifies detected objects into predefined categories.
- **Object Counting**: Counts the number of each type of object.
- **Output Generation**: Generates output images with bounding boxes, labels, and counts, and prints a summary report.

## Requirements
- Python 3.x
- OpenCV
- NumPy
- tqdm
- Matplotlib

## Setup
1. **Clone the repository**:
```
git clone https://github.com/yourusername/object-counting-classification.git
cd object-counting-classification
```

2. **Install the required packages**:
```
pip install -r requirements.txt
```

3. **Download YOLO weights and configuration files**:
    - Download the YOLOv3 weights from [here](https://pjreddie.com/media/files/yolov3.weights) and place it in the `models` directory.
    - Ensure `yolov3.cfg` and `coco.names` are also in the `models` directory.

4. **Prepare the dataset**:
    - Place your test images in the `data/test_images/val2017` directory.

## Usage
1. **Run the main script**:
```
python src/main.py
```


2. **Output**:
    - Processed images with bounding boxes, labels, and counts will be saved in the `data/output_images` directory.
    - A JSON file with the counts will be saved as `data/predicted_counts.json`.
    - A summary report will be printed to the console.

## Code Structure
- `src/preprocess.py`: Contains the image pre-processing logic.
- `src/detect.py`: Contains the YOLO model loading and object detection logic.
- `src/classify.py`: Contains the object classification logic.
- `src/count.py`: Contains the object counting logic.
- `src/output.py`: Contains functions to draw output and generate reports.
- `src/main.py`: Main script to run the entire pipeline.
- `src/test_preprocessing.py`: Script to visualize the pre-processing steps.
- `src/test_detection.py`: Script to visualize the object detection.
- `src/test_classification.py`: Script to visualize the object classification.

## Approach
1. **Image Pre-processing**: Convert images to grayscale, apply Gaussian blur for noise reduction, and enhance contrast using histogram equalization.
2. **Object Detection**: Use YOLOv3 to detect objects in the pre-processed images.
3. **Object Classification**: Classify detected objects into predefined categories (person, bicycle, car).
4. **Object Counting**: Count the number of each type of object.
5. **Output Generation**: Draw bounding boxes, labels, and counts on the images and save them. Generate a summary report of the counts.

## Assumptions and Limitations
- The system is designed to detect and classify only three types of objects: persons, bicycles, and cars.
- The YOLO model used is pre-trained on the COCO dataset, which may not perform well on images with objects not present in the COCO dataset.
- The confidence threshold for object detection is set to 0.5, which may need adjustment based on the dataset.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For any questions or issues, please contact [keithtyser@gmail.com](mailto:keithtyser@gmail.com).