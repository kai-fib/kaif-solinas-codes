from numpy import expand_dims
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import os
from os.path import isfile, join

input_path = 'C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/dump/trial/'  # Input folder
output_path = 'C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/dump/trial/done1/'  # Output folder for augmented images

if not os.path.exists(output_path):
    os.makedirs(output_path)

onlyfiles = [f for f in os.listdir(input_path) if isfile(join(input_path, f))]

def enhance_texture(image):
    """Enhance the texture by increasing contrast and applying sharpening."""
    # Convert to grayscale for better texture enhancement
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    
    # Convert back to 3-channel image
    enhanced_image = cv2.merge([enhanced_gray] * 3)
    
    # Apply sharpening filter
    kernel = np.array([[0, -0.5, 0], [-0.5, 4.5, -0.5],[0, -0.5, 0]])
    sharpened = cv2.filter2D(enhanced_image, -1, kernel)
    
    return sharpened

for n in range(len(onlyfiles)):
    input_image_path = join(input_path, onlyfiles[n])
    
    # Load and prepare the image
    img = load_img(input_image_path)
    data = img_to_array(img)
    samples = expand_dims(data, 0)

    # Image augmentation without rotation or blur
    datagen = ImageDataGenerator(
        brightness_range=[0.8, 1.2],  # Vary brightness slightly
        fill_mode='nearest'
    )
    
    it = datagen.flow(samples, batch_size=1)
    
    for i in range(5):  # Generate 5 augmented images per input
        batch = next(it)
        image = batch[0].astype('uint8')
        
        # Apply texture enhancement
        enhanced_image = enhance_texture(image)
        
        output_image_path = f'{output_path}Roots_{n}_{i+1:04}.jpg'
        cv2.imwrite(output_image_path, cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
        print(f"Saved: {output_image_path}")