# from PIL import Image, ImageDraw
# import matplotlib.pyplot as plt
# import numpy as np

# source_img = r"C:\Users\Kaif Ibrahim\Desktop\solinas_downloads\misc\sample_1.png"

# image = Image.open(source_img)
# width, height = image.size
# centre_x,centre_y = width //2, height//2



# draw = ImageDraw.Draw(image)
# draw.line([(centre_x, 0), (centre_x, height)], fill="red", width=2) 
# draw.line([(0, centre_y), (width, centre_y)], fill="red", width=2)  

# plt.imshow(image)
# plt.show()

# import math
# x1,y1,x2,y2=661, 451, 789, 545, 0 

# xo,yo=640, 360

# bx = x1+x2//2

# by = y1+y2//2



# relative_x = 192.5
# relative_y = 65.5

# r = math.sqrt(relative_x**2 + relative_y**2)  # √(192.5² + 65.5²)
# theta_rad = math.atan2(relative_y, relative_x)
# theta_deg = math.degrees(theta_rad)

# print(theta_deg)

# r ≈ sqrt(37056.25 + 4290.25) = sqrt(41346.5) ≈ 203.34
# θ ≈ atan2(-65.5, 192.5) ≈ -18.7°  (below the x-axis)


# from PIL import Image, ImageDraw
# import matplotlib.pyplot as plt
# import numpy as np
# import math

# # Load image
# source_img = r"C:\Users\Kaif Ibrahim\Desktop\solinas_downloads\misc\sample_1.png"
# image = Image.open(source_img)
# width, height = image.size
# centre_x, centre_y = width // 2, height // 2

# # Max radius to touch image corners
# radius = int(math.hypot(width, height))  # diagonal length

# # Draw clock face
# draw = ImageDraw.Draw(image)


# draw.line([(centre_x, 0), (centre_x, height)], fill="red", width=2) 
# draw.line([(0, centre_y), (width, centre_y)], fill="red", width=2)  


# for i in range(12):
#     angle_deg = (90 - i * 30) % 360  
#     angle_rad = math.radians(angle_deg)

#     end_x = centre_x + radius * math.cos(angle_rad)
#     end_y = centre_y - radius * math.sin(angle_rad)  

#     draw.line([(centre_x, centre_y), (end_x, end_y)], fill="blue", width=2)

# # Show result
# plt.figure(figsize=(10, 6))
# plt.imshow(image)
# plt.show()


"""angle"""
# import math

# def bbox_to_clock_label(bbox, image_center):
#     x1, y1, x2, y2, _ = bbox
#     cx, cy = image_center

#     # Center of bbox
#     bx = (x1 + x2) / 2
#     by = (y1 + y2) / 2

#     # Relative position to image center
#     dx = bx - cx
#     dy = cy - by  # invert y for clock (top is 12)

#     # Compute angle in degrees
#     theta_rad = math.atan2(dy, dx)
#     theta_deg = math.degrees(theta_rad)
#     print(theta_deg)
 
#     # Convert to clock angle (0° at top, increasing clockwise)
#     clock_angle = (90 - theta_deg) % 360
#     print(clock_angle)

#     # Map to 12-hour clock (each hour = 30°)
#     clock_hour = int(round(clock_angle / 30)) % 12
#     clock_hour = 12 if clock_hour == 0 else clock_hour


#     return f"{clock_hour} o'clock"

 


# bbox =[2, 233, 1192, 725, 0]
# centre = (607, 457)
# k=bbox_to_clock_label(bbox,centre)
# print(k)


# import cv2
# import numpy as np

# def hex_to_bgr(hex_color):
#     """Convert hex color code to BGR tuple."""
#     return tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))

# def clock_direction_from_color(image, hex_color, tolerance=40):
#     bgr_color = hex_to_bgr(hex_color)

#     # Define lower and upper bounds for color detection
#     lower = np.array([max(c - tolerance, 0) for c in bgr_color], dtype=np.uint8)
#     upper = np.array([min(c + tolerance, 255) for c in bgr_color], dtype=np.uint8)

#     # Create mask where color matches
#     mask = cv2.inRange(image, lower, upper)

#     # Find contours
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         return None, None

#     largest = max(contours, key=cv2.contourArea)
#     M = cv2.moments(largest)
#     if M["m00"] == 0:
#         return None, None
#     cx = int(M["m10"] / M["m00"])
#     cy = int(M["m01"] / M["m00"])

#     # Get center of the image
#     h, w = image.shape[:2]
#     center_x, center_y = w // 2, h // 2

#     # Vector from center to object
#     dx = cx - center_x
#     dy = center_y - cy  # Invert Y to make top = 0°

#     theta_deg = np.degrees(np.arctan2(dy, dx))
#     clock_angle = (90 - theta_deg) % 360
#     clock_hour = int(round(clock_angle / 30)) % 12
#     clock_hour = 12 if clock_hour == 0 else clock_hour

#     return clock_hour, clock_angle

# # Load the image and apply function
# image_path = "/mnt/data/sample_1.png"
# image = cv2.imread(image_path)
# clock_hour, clock_angle = clock_direction_from_color(image, "#0000FF")
# clock_hour, clock_angle



# import cv2
# import numpy as np

# def hex_to_bgr(hex_color):
#     return tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))

# def detect_clock_from_known_colors(image, hex_colors, tolerance=40):
#     h, w = image.shape[:2]
#     center_x, center_y = w // 2, h // 2
#     detected = []

#     for hex_color in hex_colors:
#         bgr_color = hex_to_bgr(hex_color)

#         lower = np.array([max(c - tolerance, 0) for c in bgr_color], dtype=np.uint8)
#         upper = np.array([min(c + tolerance, 255) for c in bgr_color], dtype=np.uint8)

#         mask = cv2.inRange(image, lower, upper)
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         if contours:
#             largest = max(contours, key=cv2.contourArea)
#             M = cv2.moments(largest)
#             if M["m00"] == 0:
#                 continue
#             cx = int(M["m10"] / M["m00"])
#             cy = int(M["m01"] / M["m00"])

#             dx = cx - center_x
#             dy = center_y - cy  # Flip y

#             theta_deg = np.degrees(np.arctan2(dy, dx))
#             clock_angle = (90 - theta_deg) % 360
#             clock_hour = int(round(clock_angle / 30)) % 12
#             clock_hour = 12 if clock_hour == 0 else clock_hour

#             detected.append((hex_color, clock_hour, round(clock_angle, 2)))

#     return detected

# # Load image
# image = cv2.imread("C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/test/kaif/test8/mid/Fracture(F,4)_(0-0-0)_(005.1m).png")

# # Define color codes to search
# hex_colors = ["#0000FF", "#00FF00", "#FFF36D", "#FFB81C", "#FF0000"]

# # Detect
# results = detect_clock_from_known_colors(image, hex_colors)

# # Print results
# for color, hour, angle in results:
#     print(f"Color {color} → Clock position: {hour} o'clock (angle: {angle}°)")


import cv2
import numpy as np
import math

def hex_to_bgr(hex_code):
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16) for i in (4, 2, 0))  # to BGR

def clock_direction_from_color(image, hex_code):
    # Convert hex to BGR
    target_bgr = np.array(hex_to_bgr(hex_code), dtype=np.uint8)

    # Create mask for target color (with tolerance)
    lower = np.clip(target_bgr - 20, 0, 255)
    upper = np.clip(target_bgr + 20, 0, 255)
    
    mask = cv2.inRange(image, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None

    # Take largest contour (assuming it's the bbox)
    contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(contour)
    
    if M['m00'] == 0:
        return None, None

    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # Image center
    h, w = image.shape[:2]
    img_cx, img_cy = w // 2, h // 2

    dx = cx - img_cx
    dy = img_cy - cy  # y flipped for clock

    angle_rad = math.atan2(dx, dy)
    angle_deg = (math.degrees(angle_rad) + 360) % 360

    clock_hour = int((angle_deg + 15) // 30) % 12
    clock_hour = 12 if clock_hour == 0 else clock_hour

    return clock_hour, angle_deg


image = cv2.imread(r"C:\Users\Kaif Ibrahim\Desktop\solinas_downloads\test\kaif\test8\mid\Fracture(F,4)_(0-0-0)_(005.1m).png")
clock_hour, clock_angle = clock_direction_from_color(image, "#0000FF")

if clock_hour:
    print(f"Defect is at {clock_hour} o'clock (angle: {clock_angle:.2f}°)")
else:
    print("Defect color not found in the image.")









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
