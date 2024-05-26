# src/preprocess.py
import cv2

def preprocess_image(image):
    """
    Pre-process the input image by applying noise reduction and contrast enhancement.
    
    Parameters:
    - image: Input image in BGR format.
    
    Returns:
    - enhanced: Pre-processed image.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply contrast enhancement using histogram equalization
    enhanced_gray = cv2.equalizeHist(blurred)
    
    # Convert back to three channels
    enhanced = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
    
    return enhanced
