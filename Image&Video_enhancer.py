# """using video"""
# import cv2
# import numpy as np

# # Function to apply unblurring, brightness, and contrast adjustment to the frame
# def enhance_frame(frame, brightness=0, contrast=1.2):
#     # Create a kernel for sharpening
#     kernel = np.array([[0, -1, 0],
#                        [-1, 5, -1],
#                        [0, -1, 0]])
    
#     # Apply sharpening filter to unblur
#     unblurred = cv2.filter2D(frame, -1, kernel)
    
#     # Increase brightness and contrast
#     enhanced = cv2.convertScaleAbs(unblurred, alpha=contrast, beta=brightness)
    
#     return enhanced

# # Path to input video and output video
# input_video_path = 'D:/SOLINAS DOWNLOADS/dead/try/koif/hyundai.mp4'
# output_video_path = 'D:/SOLINAS DOWNLOADS/dead/try/koif/hyundai_2.mp4'

# # Open the video
# cap = cv2.VideoCapture(input_video_path)

# # Get video properties
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS)
# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# # Define the codec and create VideoWriter object
# out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# # Process the video frame by frame
# frame_num = 0
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     # Enhance the current frame
#     enhanced_frame = enhance_frame(frame, brightness=30, contrast=1.2)
    
#     # Write the processed frame to the output video
#     out.write(enhanced_frame)
    
#     frame_num += 1
#     print(f'Processing frame {frame_num}/{total_frames}', end='\r')

# # Release everything when the job is finished
# cap.release()
# out.release()
# cv2.destroyAllWindows()

# print('Video processing complete!')






# import cv2
# import numpy as np

# def enhance_image(img):
#     # Step 1: Sharpen the image with a milder kernel
#     mild_sharpening_kernel = np.array([[0, -0.5, 0],
#                                        [-0.5, 3,-0.5],
#                                        [0, -0.5, 0]])
    
#     # Apply mild sharpening filter
#     sharpened = cv2.filter2D(img, -1, mild_sharpening_kernel)
    
#     # Step 2: Moderate contrast enhancement using CLAHE
#     lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
    
#     # Apply CLAHE with reduced clipLimit to avoid over-enhancement
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     cl = clahe.apply(l)
    
#     limg = cv2.merge((cl, a, b))
#     contrast_enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

#     # Step 3: Slight unsharp masking (milder than before)
#     gaussian = cv2.GaussianBlur(contrast_enhanced, (9,9), 10.0)
#     unsharp_image = cv2.addWeighted(contrast_enhanced, 1.2, gaussian, -0.2, 0)

#     return unsharp_image

# # Load the image
# image_path = "D:/Gugan/frame_1490.jpg"
# img = cv2.imread(image_path)

# # Enhance the image with moderate settings
# enhanced_img = enhance_image(img)

# # Save the enhanced image
# output_path = 'C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/dump/'
# cv2.imwrite(output_path, enhanced_img)

# # Display the image (optional)
# cv2.imshow('Moderately Enhanced Image', enhanced_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print(f"Moderately enhanced image saved to {output_path}")



import cv2
import numpy as np
import os

def enhance_image(img):
    if img is None:
        print("Error: Image not loaded properly.")
        return None

    # Step 1: Sharpen the image with a milder kernel
    mild_sharpening_kernel = np.array([[0, -0.5, 0],
                                       [-0.5, 3, -0.5],
                                       [0, -0.5, 0]])
    
    # Apply mild sharpening filter
    sharpened = cv2.filter2D(img, -1, mild_sharpening_kernel)
    
    # Step 2: Moderate contrast enhancement using CLAHE
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE with reduced clipLimit to avoid over-enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl, a, b))
    contrast_enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Step 3: Slight unsharp masking (milder than before)
    gaussian = cv2.GaussianBlur(contrast_enhanced, (9,9), 10.0)
    unsharp_image = cv2.addWeighted(contrast_enhanced, 1.2, gaussian, -0.2, 0)

    return unsharp_image

# Input and Output Paths
input_dir = "D:/Gugan/OG/"
output_dir = "C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/dump/Enhanced/"
os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

# Ensure input directory exists
if not os.path.exists(input_dir):
    print(f"Error: Directory not found at {input_dir}")
    exit()

# Process Each Image in the Directory
img_list = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not img_list:
    print("No images found in the input directory.")
    exit()

for img_name in img_list:
    img_path = os.path.join(input_dir, img_name)  # Full image path
    img = cv2.imread(img_path)

    if img is None:
        print(f"Error: Could not read {img_path}")
        continue

    # Enhance the image
    enhanced_img = enhance_image(img)

    if enhanced_img is not None:
        output_path = os.path.join(output_dir, f"enhanced_{img_name}")  # Unique name

        if cv2.imwrite(output_path, enhanced_img):
            print(f"Enhanced image saved: {output_path}")
        else:
            print(f"Error: Could not save {output_path}")

        # Optional Display (Uncomment if you want to see the images)
        # cv2.imshow('Enhanced Image', enhanced_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    else:
        print(f"Error: Enhancement failed for {img_name}")

