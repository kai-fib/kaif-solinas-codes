import cv2
import numpy as np
# # Load image
# img = cv2.imread(r"D:\kai\diff.png")
# # Mouse callback function
# def show_pixel_value(event, x, y, flags, param):
#     if event == cv2.EVENT_MOUSEMOVE:  # when mouse moves
#         pixel = img[y, x]  # BGR format
#         print(f"X:{x}, Y:{y} -> Pixel Value: {pixel}")
# # Create window and set callback
# cv2.imshow("Image", img)
# cv2.setMouseCallback("Image", show_pixel_value)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Passing path of image as parameter
img = cv2.imread(r"D:\no_dist\nope.png")

# If the extension of our image was 
# '.jpg' then we have passed it as 
# argument instead of '.png'.
img_encode = cv2.imencode('.png', img)[1]

# Converting the image into numpy array
data_encode = np.array(img_encode)

# Converting the array to bytes.
byte_encode = data_encode.tobytes()

print(byte_encode)