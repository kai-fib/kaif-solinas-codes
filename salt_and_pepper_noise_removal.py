# import cv2
# from os import listdir

# S = listdir("D:/SOLINAS DOWNLOADS/Dataset to be Given/ferrule and fracture/fracture/New folder/salt/")
# outpath = "D:/SOLINAS DOWNLOADS/Dataset to be Given/ferrule and fracture/fracture/New folder/salt/no_salt"

# for i in range(0, len(S)):
#     img = cv2.imread(S[i])
#     img_filtered = cv2.medianBlur(img, 3)
#     cv2.imwrite(outpath + S[i], img_filtered)



import cv2
from os import listdir
import os


input_dir = "C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/fracture_final/"
#output_dir = "C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/fracture_final/n/"


# List all files in the directory
S = listdir(input_dir)

for i in range(0, len(S)):
    img_path = os.path.join(input_dir, S[i])
    img = cv2.imread(img_path)

    # Check if the image was loaded successfully
    if img is None:
        print(f"Failed to load image {img_path}")
        continue

    img_filtered = cv2.medianBlur(img, 3)
    #output_path = os.path.join(output_dir, S[i])
    cv2.imwrite(img_path, img_filtered)#img_filtered
    print(f"Processed and saved {img_path}")




# import cv2
# import numpy as np

# # Load the image with salt and pepper noise
# image_path = 'Fracture_00386.jpg'
# img = cv2.imread(image_path, cv2.IMREAD_COLOR)

# # Define a kernel
# kernel = np.ones((3, 3), np.uint8)

# # Apply morphological opening
# opened_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# # Display the original and processed images
# cv2.imshow('Original Image', img)
# cv2.imshow('Opened Image', opened_img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
