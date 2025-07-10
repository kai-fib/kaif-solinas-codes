# #import rembg
# from rembg import remove
# from PIL import Image
# import os
# from os import listdir
# # Store path of the image in the variable input_path
# input_path =  'C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/3D_imp/session_2/Endo90/set5_(kaif)/'
# # Store path of the output image in the variable output_path
# output_path = 'C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/3D_imp/session_2/Endo90/set5_(kaif)/out_bgr/'
# A = listdir(input_path)
# for i in range(0,len(input_path)):
#   # Processing the image
#   input = Image.open(input_path + A[i])
#   # Removing the background from the given Image
#   output = remove(input)
#   new_file = os.path.splitext(A[i])[0] + '.jpg'
#   #Saving the image in the given path
#   output.save(output_path + new_file )





# from rembg import remove
# from PIL import Image
# import os

# # Store paths
# input_path = 'C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/3D_imp/session_2/Endo90/set5_(kaif)/'
# output_path = 'C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/3D_imp/session_2/Endo90/set5_(kaif)/out_bgr/'

# # Ensure output directory exists
# os.makedirs(output_path, exist_ok=True)

# # Get list of files
# A = os.listdir(input_path)

# # Loop through all files
# for i in range(len(A)):
#     # Only process image files
#     if not A[i].lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
#         continue
    
#     input_file = os.path.join(input_path, A[i])
#     output_file = os.path.join(output_path, os.path.splitext(A[i])[0] + '.jpg')

#     try:
#         # Open image
#         input_img = Image.open(input_file)

#         # Remove background
#         output_img = remove(input_img)

#         # Convert RGBA to RGB (JPEG does not support transparency)
#         output_img.convert("RGB").save(output_file, "JPEG")

#         print(f"Processed: {A[i]}")
    
#     except Exception as e:
#         print(f"Error processing {A[i]}: {e}")




import cv2
import numpy as np

# Load image
image = cv2.imread("C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/3D_imp/session_2/Endo90/set5_(kaif)/DJI_0446.JPG")
mask = np.zeros(image.shape[:2], np.uint8)

# Define background and foreground models
bgdModel = np.zeros((1,65), np.float64)
fgdModel = np.zeros((1,65), np.float64)

# Define a rectangle around the robot
rect = (50, 50, image.shape[1]-50, image.shape[0]-50)

# Apply GrabCut algorithm
cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# Convert mask to binary
mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
result = image * mask2[:, :, np.newaxis]

# Save the mask
cv2.imwrite("C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/DJI_0446_mask.png", mask2 * 255)
