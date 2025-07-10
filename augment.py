"""This is the augment code using TensorFlow"""
from numpy import expand_dims
from tensorflow.keras.utils import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
import cv2
import os
from os.path import isfile, join


input_path = "C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/Video_dataset/Revamp-V9/water_2.1.0/images/ad_sewer/"  # Input folder
output_path = 'C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/Video_dataset/Revamp-V9/water_2.1.0/images/ad_sewer/aug_ad/'  # Output folder for augmented images

valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
onlyfiles = [
    f for f in os.listdir(input_path)
    if isfile(join(input_path, f)) and f.lower().endswith(valid_exts)
]

for n in range(len(onlyfiles)):
    onlyfiles[n]
    input_image_path = join(input_path, onlyfiles[n])

    # Load and prepare the image
    #img = load_img(input_image_path, target_size=(224, 224))
    img = load_img(input_image_path)
    data = img_to_array(img)
    samples = expand_dims(data, 0)

    datagen = ImageDataGenerator(fill_mode='constant',brightness_range=[0.7,1.0],shear_range=0.3)

    
    it = datagen.flow(samples, batch_size=1)

    #koif make 5 aug image
    for i in range(6):
        batch = next(it)
        image = batch[0].astype('uint8')
        
        
        output_image_path = f'{output_path}Back_{n}_{i+1:04}.jpg'
        cv2.imwrite(output_image_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        print(f"Saved: {output_image_path}")
