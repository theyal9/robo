# preprocess.py
import cv2
import numpy as np

def preprocess_image(img_path):
    # Read image from the path
    img = cv2.imread(img_path)
    # Resize image to 128x128 for consistency
    resized = cv2.resize(img, (128, 128))
    # Normalize image pixels
    normalized = resized / 255.0
    return normalized

def find_card_contours(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply GaussianBlur to reduce noise and improve contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Detect edges using Canny edge detection
    edged = cv2.Canny(blurred, 50, 150)
    # Find contours
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Approximate the contour and check if it's rectangular
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            card_img = img[y:y+h, x:x+w]
            return card_img
    return None
