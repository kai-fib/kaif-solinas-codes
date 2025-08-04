# """using video"""
# # import cv2
# # import numpy as np

# # # Function to apply unblurring, brightness, and contrast adjustment to the frame
# # def enhance_frame(frame, brightness=0, contrast=1.2):
# #     # Create a kernel for sharpening
# #     kernel = np.array([[0, -1, 0],
# #                        [-1, 5, -1],
# #                        [0, -1, 0]])
    
# #     # Apply sharpening filter to unblur
# #     unblurred = cv2.filter2D(frame, -1, kernel)
    
# #     # Increase brightness and contrast
# #     enhanced = cv2.convertScaleAbs(unblurred, alpha=contrast, beta=brightness)
    
# #     return enhanced

# # # Path to input video and output video
# # input_video_path = 'D:/SOLINAS DOWNLOADS/dead/try/koif/hyundai.mp4'
# # output_video_path = 'D:/SOLINAS DOWNLOADS/dead/try/koif/hyundai_2.mp4'

# # # Open the video
# # cap = cv2.VideoCapture(input_video_path)

# # # Get video properties
# # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# # fps = cap.get(cv2.CAP_PROP_FPS)
# # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# # # Define the codec and create VideoWriter object
# # out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# # # Process the video frame by frame
# # frame_num = 0
# # while cap.isOpened():
# #     ret, frame = cap.read()
# #     if not ret:
# #         break
    
# #     # Enhance the current frame
# #     enhanced_frame = enhance_frame(frame, brightness=30, contrast=1.2)
    
# #     # Write the processed frame to the output video
# #     out.write(enhanced_frame)
    
# #     frame_num += 1
# #     print(f'Processing frame {frame_num}/{total_frames}', end='\r')

# # # Release everything when the job is finished
# # cap.release()
# # out.release()
# # cv2.destroyAllWindows()

# # print('Video processing complete!')






# # import cv2
# # import numpy as np

# # def enhance_image(img):
# #     # Step 1: Sharpen the image with a milder kernel
# #     mild_sharpening_kernel = np.array([[0, -0.5, 0],
# #                                        [-0.5, 3,-0.5],
# #                                        [0, -0.5, 0]])
    
# #     # Apply mild sharpening filter
# #     sharpened = cv2.filter2D(img, -1, mild_sharpening_kernel)
    
# #     # Step 2: Moderate contrast enhancement using CLAHE
# #     lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
# #     l, a, b = cv2.split(lab)
    
# #     # Apply CLAHE with reduced clipLimit to avoid over-enhancement
# #     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# #     cl = clahe.apply(l)
    
# #     limg = cv2.merge((cl, a, b))
# #     contrast_enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

# #     # Step 3: Slight unsharp masking (milder than before)
# #     gaussian = cv2.GaussianBlur(contrast_enhanced, (9,9), 10.0)
# #     unsharp_image = cv2.addWeighted(contrast_enhanced, 1.2, gaussian, -0.2, 0)

# #     return unsharp_image

# # # Load the image
# # image_path = 'E:/chettos/kaif.jpg'
# # img = cv2.imread(image_path)

# # # Enhance the image with moderate settings
# # enhanced_img = enhance_image(img)

# # # Save the enhanced image
# # output_path = 'E:/chettos/task/'
# # cv2.imwrite(output_path, enhanced_img)

# # # Display the image (optional)
# # cv2.imshow('Moderately Enhanced Image', enhanced_img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# # print(f"Moderately enhanced image saved to {output_path}")





import cv2
import numpy as np
import os
from PIL import Image


input_folder = "C:/Users/Kaif Ibrahim/Desktop/chettos/again/"
output_folder = "C:/Users/Kaif Ibrahim/Desktop/chettos/again/task/ok"


if not os.path.exists(output_folder):
    os.makedirs(output_folder)


def process_image(image_path, output_path):
    
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    kernel = np.array([[0, -1, 0],
                       [-1,  5, -1],
                       [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(img_rgb, -1, kernel)

    # Enhance contrast using CLAHE
    lab = cv2.cvtColor(sharpened, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_enhanced = cv2.merge((l_clahe, a, b))
    img_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    # Adjust brightness and contrast
    contrast = 1.2  # Increase contrast slightly
    brightness = 10  # Brightness offset
    final_img = cv2.convertScaleAbs(img_enhanced, alpha=contrast, beta=brightness)

    Image.fromarray(final_img).save(output_path)


for file_name in os.listdir(input_folder):
    if file_name.lower().endswith(".jpg"):
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)
        print(f"Processing {file_name}...")
        process_image(input_path, output_path)

print("koif its here:", output_folder)


























# # import cv2
# # import numpy as np
# # import os
# # from PIL import Image

# # # Path to the folder containing the JPEG files
# # input_folder = "C:/Users/Kaif Ibrahim/Desktop/chettos/again/"  # Replace with your folder path
# # output_folder = "C:/Users/Kaif Ibrahim/Desktop/chettos/again/task"   # Folder to save processed images

# # # Create the output folder if it doesn't exist
# # os.makedirs(output_folder, exist_ok=True)

# # # Function to process a single image
# # def process_image(image_path, output_path):
# #     # Load the image
# #     cv_img = cv2.imread(image_path)

# #     # Convert the image from BGR to RGB (OpenCV loads images in BGR format)
# #     cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

# #     # Step 1: Apply sharpening filter to emphasize details
# #     sharpening_kernel = np.array([
# #         [0, -1, 0],
# #         [-1, 5, -1],
# #         [0, -1, 0]
# #     ], dtype=np.float32)
# #     sharpened_img = cv2.filter2D(cv_img_rgb, -1, sharpening_kernel)

# #     # Step 2: Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
# #     lab_img = cv2.cvtColor(sharpened_img, cv2.COLOR_RGB2LAB)  # Convert to LAB color space
# #     l, a, b = cv2.split(lab_img)
# #     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
# #     l_clahe = clahe.apply(l)
# #     enhanced_lab = cv2.merge((l_clahe, a, b))
# #     enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)  # Convert back to RGB

# #     # Step 3: Adjust brightness and contrast globally (fine-tuning)
# #     alpha = 1.2  # Contrast factor (1.0 means no change)
# #     beta = 20    # Brightness factor (0 means no change)
# #     final_img = cv2.convertScaleAbs(enhanced_img, alpha=alpha, beta=beta)

# #     # Save the processed image
# #     final_img_pil = Image.fromarray(final_img)
# #     final_img_pil.save(output_path)

# # # Loop through all JPEG files in the folder
# # for filename in os.listdir(input_folder):
# #     if filename.lower().endswith(".jpg"):  # Check for JPEG files
# #         input_path = os.path.join(input_folder, filename)
# #         output_path = os.path.join(output_folder, filename)
        
# #         print(f"Processing: {filename}")
# #         process_image(input_path, output_path)

# # print("Processing complete! All enhanced images are saved in:", output_folder)

# from PIL import Image, ImageEnhance

# # Load the image
# image = Image.open('C:/Users/Kaif Ibrahim/Desktop/chettos/again/frame_23866.jpg')

# # Create an enhancer object
# enhancer = ImageEnhance.Contrast(image)

# # Apply the enhancement
# factor = 1.5  # Increase contrast
# image_enhanced = enhancer.enhance(factor)

# # Save the enhanced image
# image_enhanced.save('C:/Users/Kaif Ibrahim/Desktop/chettos/