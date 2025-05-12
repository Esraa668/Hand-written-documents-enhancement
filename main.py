import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

def enhance_document(image_path, output_dir="output"):

    os.makedirs(output_dir, exist_ok=True)

    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) #grayscale conversion
    if original is None:
        raise ValueError(f"Image not found at {image_path}") #checks if image exists in the destined path
 #sharpening to reduce blurness using laplacian sharpening
    
    laplacian = cv2.Laplacian(original, cv2.CV_64F) #Laplacian is faster and simpler for generic edge detection /CV_64F is used to allow negative values ).
    sharpened = cv2.convertScaleAbs(original - 0.5 * laplacian)
