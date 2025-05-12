import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

def enhance_document(image_path, output_dir="output"):

    os.makedirs(output_dir, exist_ok=True)

    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) #grayscale conversion
    if original is None:
        raise ValueError(f"Image not found at {image_path}") #checks if image exists in the destined path
