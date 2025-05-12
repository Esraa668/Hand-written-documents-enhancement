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

    #Contrast enhancement using percentile clipping smarter form of contrast stretching
    low, high = np.percentile(sharpened, (2, 98))
    stretched = np.clip(sharpened, low, high)
    enhanced = cv2.normalize(stretched, None, 0, 255, cv2.NORM_MINMAX)

    #morphological closing to reconnect broken strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    final = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel, iterations=1)

    
    filename = os.path.splitext(os.path.basename(image_path))[0] #Extracts the name of the file without its extension.
    cv2.imwrite(f"{output_dir}/{filename}_enhanced.png", final) #automatically name the output filename based on the input

    #Display original and enhanced together for comparison
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1), plt.imshow(original, cmap='gray'), plt.title("Original")
    plt.subplot(1, 2, 2), plt.imshow(final, cmap='gray'), plt.title("Enhanced")
    plt.tight_layout()
    plt.show()

    return original, final

if _name_ == "_main_":
    img_path = "C:/Users/DELL/Downloads/PythonProject/doccc.png"
    enhance_document(img_path)