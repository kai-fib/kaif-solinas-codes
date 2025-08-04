"""Intensity of an Image at a particular point hovering mouse"""
# import cv2

# # Load image
# img = cv2.imread("C:/Users/Kaif Ibrahim/Downloads/003 - 03.13.41 PM.jpeg")  # Replace with your image path

# # Mouse callback function
# def show_intensity(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         pixel = img[y, x]
#         if len(pixel.shape) == 0 or len(pixel) == 1:  # Grayscale
#             print(f"Clicked at ({x}, {y}) - Intensity: {pixel}")
#         else:  # Color image
#             print(f"Clicked at ({x}, {y}) - BGR: {pixel} | Intensity: {int(sum(pixel)/3)}")

# # Show image window
# cv2.imshow("Image", img)
# cv2.setMouseCallback("Image", show_intensity)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import math

# def find_defect_clock_position(x_min, y_min, x_max, y_max, center_x, center_y):
#     # Get the 4 corner points of the bounding box
#     # if defect == "Settled_Deposit" or ""
#     corners = [
#         (x_min, y_min),  # top-left
#         (x_max, y_min),  # top-right
#         (x_min, y_max),  # bottom-left
#         (x_max, y_max)   # bottom-right
#     ]

#     clock_hours = []

#     for x, y in corners:
#         # Calculate difference from center
#         dx = x - center_x
#         dy = center_y - y  # Flip Y because image Y grows down

#         # Get angle in radians and convert to degrees
#         angle_rad = math.atan2(dy, dx)
#         angle_deg = (math.degrees(angle_rad) + 360) % 360  # Make sure angle is 0-360

#         # Convert angle to clock hour (30° per hour, shift by 15° to center bins)
#         hour = int((angle_deg + 15) // 30) % 12
#         if hour == 0:
#             hour = 12

#         clock_hours.append(hour)

#     # Keep only unique hours and sort them
#     unique_hours = sorted(set(clock_hours))

#     # Check if defect crosses 12 o'clock (e.g. from 11 to 1)
#     if len(unique_hours) > 1 and (unique_hours[-1] - unique_hours[0] > 6):
#         early = [h for h in unique_hours if h < 6]
#         late = [h for h in unique_hours if h >= 6]
#         if early and late:
#             return f"Defect spans from {late[0]} to {early[-1]} o'clock"

#     # Normal cases
#     if len(unique_hours) == 1:
#         return f"Defect is located at {unique_hours[0]} o'clock"
#     else:
#         return f"Defect spans from {unique_hours[0]} to {unique_hours[-1]} o'clock"


# print(find_defect_clock_position(1033, 171, 1279, 563, 640, 360))

# import math

# def get_angle_deg(x, y, cx, cy):
#     """Compute angle from image center to point (x, y), adjusted for image coordinate system."""
#     dx = x - cx
#     dy = cy - y  # Flip because image y increases downward
#     angle_rad = math.atan2(dy, dx)
#     return (math.degrees(angle_rad) + 360) % 360  # Normalize to [0, 360)

# def angle_to_clock_hour(angle):
#     """Map angle (in degrees) to clock hour (1 to 12)."""
#     return int((angle + 15) // 30) % 12 or 12  # each hour = 30°

# def get_defect_clock_range(x_min, y_min, x_max, y_max, cx, cy):
#     """Return clock hour range for a defect bounding box."""
#     # Define all 4 corners of the bounding box
#     corners = [
#         (x_min, y_min),  # top-left
#         (x_max, y_min),  # top-right
#         (x_min, y_max),  # bottom-left
#         (x_max, y_max),  # bottom-right
#     ]

    
#     # Get unique clock hours for each corner
#     clock_hours = sorted({angle_to_clock_hour(get_angle_deg(x, y, cx, cy)) for (x, y) in corners})

#     # Handle wrap-around (e.g., 11 → 12 → 1 → 2)
#     if len(clock_hours) > 1 and (clock_hours[-1] - clock_hours[0] > 6):
#         # Clock wraps around — split into two ranges and join
#         left = [h for h in clock_hours if h < 6]     # 1 to 5
#         right = [h for h in clock_hours if h >= 6]   # 6 to 12
#         if left and right:
#             return f"Defect spans from {right[0]} to {left[-1]} o'clock"

#     # Regular case
#     if len(clock_hours) == 1:
#         return f"Defect is located at {clock_hours[0]} o'clock"
#     else:
#         return f"Defect spans from {clock_hours[0]} to {clock_hours[-1]} o'clock"



bbox = [661, 451, 789, 545]       # Example YOLO bbox
image_size = (1280, 720)           # Your image

result = find_defect_clock_position(*bbox, *image_size)
print(result)


# 661, 451, 789, 545, 640, 360

#[661, 451, 789, 545, 0], (640, 360)]
