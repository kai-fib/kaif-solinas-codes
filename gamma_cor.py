import cv2
import numpy as np
from os import listdir

def adjust_gamma(image, gamma=1.0):
    # Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # Apply gamma correction using the lookup table
    return cv2.LUT(image, table)

# Load the hazy image
# input_path = 'C:/Users/Kaif Ibrahim/Desktop/gif_chettos/new/f3/'
# S = listdir(input_path)
# outpath = "C:/Users/Kaif Ibrahim/Desktop/gif_chettos/new/f3/enh"

#for i in range(0, len(S)):
    
    #image_path = '1.jpeg'
hazy_image = cv2.imread('C:/Users/Kaif Ibrahim/Desktop/gif_chettos/new/f3/frame_0058.jpg')
    #hazy_image = cv2.imread(input_path + S[i])

    # Apply gamma correction
gamma_value = 0.55 # You can adjust this value
dehazed_image = adjust_gamma(hazy_image, gamma = gamma_value)
cv2.imwrite('C:/Users/Kaif Ibrahim/Desktop/gif_chettos/new/defect_13_1.jpg', dehazed_image)
    #cv2.imwrite(outpath + S[i], dehazed_image)
    

# Display the results
# cv2.imshow('Hazy Image', hazy_image)
# cv2.imshow('Dehazed Image', dehazed_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()