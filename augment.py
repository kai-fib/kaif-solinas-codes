"""This is the augment code using TensorFlow"""
from numpy import expand_dims
from tensorflow.keras.utils import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
import cv2
import os
from os.path import isfile, join


input_path = r"D:\Sewer_ML\consolidated_class_wise\ad"  # Input folder
output_path = r"D:\Sewer_ML\consolidated_class_wise\Augmented_data\aug_ad\ "  # Output folder for augmented images

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

    datagen = ImageDataGenerator(fill_mode='constant',shear_range=0.5,rotation_range=10,brightness_range=[0.6, 1],horizontal_flip=True) 
    #datagen = ImageDataGenerator( rotation_range=10, fill_mode='constant', brightness_range=[0.6, 1], shear_range=0.3,horizontal_flip=True )
    #datagen = ImageDataGenerator(rotation_range=7,fill_mode='constant',brightness_range=[0.6,1.3],shear_range=0.3)    
    it = datagen.flow(samples, batch_size=1)

    #koif make 5 aug image
    for i in range(5):
        batch = next(it)
        image = batch[0].astype('uint8')
        
        
        output_image_path = f'{output_path}Attached_Deposit_{n}_{i+1:04}.jpg'
        cv2.imwrite(output_image_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        print(f"Saved: {output_image_path}")
