from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

source_img = "C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/sample.jpg"

image = Image.open(source_img)
width, height = image.size
centre_x,centre_y = width //2, height//2



draw = ImageDraw.Draw(image)
draw.line([(centre_x, 0), (centre_x, height)], fill="red", width=2) 
draw.line([(0, centre_y), (width, centre_y)], fill="red", width=2)  

plt.imshow(image)
plt.show()



















# new_image = image.resize((512, 512))
# new_image.save('C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/Stage-III(Dataset)/WRC_bgrd_v2/WRC_bgrd_v2/images/image_del/SD/koif_try.jpg')

# import cv2
# import numpy as np

# def clock_region_from_bbox(image_path):
#     # Load image
#     image = cv2.imread(image_path)
#     h, w, _ = image.shape
#     cx, cy = w // 2, h // 2
#     image_area = h * w

#     # Convert image to HSV
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#     color_ranges = {
#     # Hue wraps around at 180
#     "red1":   ((0,  70, 70), (10, 255, 255)),       # Lower red
#     "red2":   ((170, 70, 70), (180, 255, 255)),     # Upper red (wraps around)

#     "blue":   ((110, 70, 70), (130, 255, 255)),     # Around hue 240° ≈ 120° OpenCV
#     "green":  ((50,  50, 50), (80, 255, 255)),      # Around hue 120°
#     "orange": ((15, 100, 100), (25, 255, 255)),     # Around hue 39°
#     "yellow": ((25, 100, 100), (35, 255, 255))      # Around hue 60°
#     }


#     # Combine all masks
#     full_mask = np.zeros_like(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
#     for key, (lower, upper) in color_ranges.items():
#         mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
#         full_mask = cv2.bitwise_or(full_mask, mask)

#     # Find contours
#     contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         return "No bounding box detected."

#     clock_sectors = set()
#     for cnt in contours:
#         x, y, bw, bh = cv2.boundingRect(cnt)
#         box_area = bw * bh
#         box_center_x = x + bw // 2
#         box_center_y = y + bh // 2

#         # Check if bounding box is large
#         if box_area > 0.6 * image_area:
#             return "Defect covers clock region: 12 to 12 (entire region)"

#         # Determine quadrant
#         if box_center_x >= cx and box_center_y < cy:
#             clock_sectors.add("12 to 3")
#         if box_center_x < cx and box_center_y < cy:
#             clock_sectors.add("9 to 12")
#         if box_center_x < cx and box_center_y >= cy:
#             clock_sectors.add("6 to 9")
#         if box_center_x >= cx and box_center_y >= cy:
#             clock_sectors.add("3 to 6")

#     if not clock_sectors:
#         return "Defect location not detected properly."

#     sorted_map = {
#         "9 to 12": 1,
#         "12 to 3": 2,
#         "3 to 6": 3,
#         "6 to 9": 4
#     }
#     sorted_clocks = sorted(clock_sectors, key=lambda x: sorted_map[x])
#     return f"Defect covers clock region: {', '.join(sorted_clocks)}"

# # Example usage
# result = clock_region_from_bbox("C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/sample_1.png")
# print(result)



# import cv2
# import numpy as np

# def clock_region_from_bbox(image_path):
#     # Load image
#     image = cv2.imread(image_path)
#     h, w, _ = image.shape
#     cx, cy = w // 2, h // 2

#     # Convert to HSV and create a mask for green (bounding box)
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     lower_green = np.array([50, 50, 50]) #(50,  50, 50) (80, 255, 255)
#     upper_green = np.array([80, 255, 255])
#     mask = cv2.inRange(hsv, lower_green, upper_green)

#     # Find contours of green regions (defect boxes)
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         return "No bounding box detected."

#     clock_sectors = set()
#     for cnt in contours:
#         x, y, bw, bh = cv2.boundingRect(cnt)
#         box_center_x = x + bw // 2
#         box_center_y = y + bh // 2

#         # Determine quadrant based on box center
#         if box_center_x >= cx and box_center_y < cy:
#             clock_sectors.add("12 to 3")
#         if box_center_x < cx and box_center_y < cy:
#             clock_sectors.add("9 to 12")
#         if box_center_x < cx and box_center_y >= cy:
#             clock_sectors.add("6 to 9")
#         if box_center_x >= cx and box_center_y >= cy:
#             clock_sectors.add("3 to 6")

#     # Combine the results
#     if not clock_sectors:
#         return "Defect location not detected properly."
#     else:
#         sorted_map = {
#             "9 to 12": 1,
#             "12 to 3": 2,
#             "3 to 6": 3,
#             "6 to 9": 4
#         }
#         sorted_clocks = sorted(clock_sectors, key=lambda x: sorted_map[x])
#         return f"Defect covers clock region: {', '.join(sorted_clocks)}"

# # Example usage
# result = clock_region_from_bbox("C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/sample_1.png")
# print(result)
