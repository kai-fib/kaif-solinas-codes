# # importing the module 
# import cv2 

# # function to display the coordinates of 
# # of the points clicked on the image 
# def click_event(event, x, y, flags, params): 

# 	# checking for left mouse clicks 
# 	if event == cv2.EVENT_LBUTTONDOWN: 

# 		# displaying the coordinates 
# 		# on the Shell 
# 		print(x, ' ', y) 

# 		# displaying the coordinates 
# 		# on the image window 
# 		font = cv2.FONT_HERSHEY_SIMPLEX 
# 		cv2.putText(img, str(x) + ',' +
# 					str(y), (x,y), font, 
# 					1, (255, 0, 0), 2) 
# 		cv2.imshow('image', img) 

# 	# checking for right mouse clicks	 
# 	if event==cv2.EVENT_RBUTTONDOWN: 

# 		# displaying the coordinates 
# 		# on the Shell 
# 		print(x, ' ', y) 

# 		# displaying the coordinates 
# 		# on the image window 
# 		font = cv2.FONT_HERSHEY_SIMPLEX 
# 		b = img[y, x, 0] 
# 		g = img[y, x, 1] 
# 		r = img[y, x, 2] 
# 		cv2.putText(img, str(b) + ',' +
# 					str(g) + ',' + str(r), 
# 					(x,y), font, 1, 
# 					(255, 255, 0), 2) 
# 		cv2.imshow('image', img) 



# 	# reading the image 
# img = cv2.imread("Circumferential_Crack_169_0003.jpg", 1) 

# 	# displaying the image 
# cv2.imshow('image', img) 

# 	# setting mouse handler for the image 
# 	# and calling the click_event() function 
# cv2.setMouseCallback('image', click_event) 

# 	# wait for a key to be pressed to exit 
# cv2.waitKey(0) 

# 	# close the window 
# cv2.destroyAllWindows() 


import numpy as np
import re

s= "mystring name is (kaif,3)"

cheese = re.findall('\(([^,]+),',s)

print(cheese)
#l =list(s)
#print(l)
# import cv2
# import os

# # Function to display the coordinates of the points clicked on the image
# def click_event(event, x, y, flags, params):
#     # Checking for left mouse clicks
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print(f"Left Click at: ({x}, {y})")
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         cv2.putText(img, f"{x},{y}", (x, y), font, 0.5, (255, 0, 0), 1)
#         cv2.imshow('Image', img)

#     # Checking for right mouse clicks
#     if event == cv2.EVENT_RBUTTONDOWN:
#         print(f"Right Click at: ({x}, {y})")
#         b, g, r = img[y, x]
#         print(f"Pixel Color (B, G, R): ({b}, {g}, {r})")
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         cv2.putText(img, f"{b},{g},{r}", (x, y), font, 0.5, (255, 255, 0), 1)
#         cv2.imshow('Image', img)

# # Main function
# if __name__ == "__main__":
#     # Input the image path dynamically
#     image_path = input("Enter the path to the image: ").strip()

#     # Check if the file exists
#     if not os.path.isfile(image_path):
#         print(f"Error: The file '{image_path}' does not exist.")
#     else:
#         # Read the image
#         img = cv2.imread(image_path)
        
#         # Check if the image was loaded successfully
#         if img is None:
#             print(f"Error: Unable to load image '{image_path}'. Please check the file path or format.")
#         else:
#             # Display the image
#             cv2.imshow('Image', img)
#             # Set mouse handler for the image window
#             cv2.setMouseCallback('Image', click_event)
#             # Wait for a key press to exit
#             cv2.waitKey(0)
#             # Close the window
#             cv2.destroyAllWindows()
