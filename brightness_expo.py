import cv2 
import numpy as np

def low_brightness_exposure(image, brightness_factor=0.4, exposure_factor=0.4):
    """
    Reduces the brightness and exposure of an image by applying a scaling factor to pixel values. 
    
    Args:
        image: The input image as a NumPy array.
        brightness_factor: Factor to adjust brightness (between 0 and 1).
        exposure_factor: Factor to adjust exposure (between 0 and 1). 
    
    Returns:
        The modified image with reduced brightness and exposure.
    """
    
    # Convert image to float for precise calculations
    image_float = image.astype(float) 
    
    # Apply brightness adjustment
    image_float = image_float * brightness_factor 
    
    # Apply exposure adjustment
    image_float = image_float * exposure_factor 
    
    # Clamp values to stay within the valid pixel range
    image_float = np.clip(image_float, 0, 255)
    
    # Convert back to integer format
    return image_float.astype(np.uint8) 

# Example usage
image = cv2.imread('C:/Users/Kaif Ibrahim/Desktop/Infil/Infilteration_9_0004.jpg')
low_brightness_image = low_brightness_exposure(image, brightness_factor=0.8, exposure_factor=0.4)
cv2.imshow('Original', image)
cv2.imshow('Low Brightness/Exposure', low_brightness_image)
cv2.waitKey(0) 
cv2.destroyAllWindows()