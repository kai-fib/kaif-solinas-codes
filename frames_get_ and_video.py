"""Normal video to frames """

import cv2
import os

# Create the output directory if it doesn't exist
output_dir = r"C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/Video_dataset/hadia/frames/"  #C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/frames/

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


cap = cv2.VideoCapture("C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/Video_dataset/hadia/133707.avi")
#F:/Sludge new/
#C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/Video_dataset/sludge/
#fps = cap.get(cv2.CAP_PROP_FPS)
#w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#print(f"FPS: {fps}")
#print(f"Width: {w}, Height: {h}")

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    ##################
    if frame_count % 5 == 0:
        resized_frame = cv2.resize(frame, (1280, 720))

         
        frame_filename = os.path.join(output_dir, f"frame_{frame_count:04}.jpg")
        cv2.imwrite(frame_filename, resized_frame)
        #continue
   


    frame_count += 1

cap.release()
cv2.destroyAllWindows()




"""continuation of one video over the  another kind of frames"""
# import cv2
# import os
# import glob

# # Define output directory
# output_dir = r"C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/merge/"
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # Function to find the last saved frame number
# def get_last_frame_number(output_dir):
#     existing_frames = glob.glob(os.path.join(output_dir, "frame_*.jpg"))
#     if not existing_frames:
#         return 0  # No previous frames exist

#     # Extract numbers from filenames (e.g., frame_018842.jpg → 18842)
#     frame_numbers = [int(os.path.basename(f).split('_')[1].split('.')[0]) for f in existing_frames]
#     return max(frame_numbers) + 1  # Start from the next number

# # Get the last frame number
# frame_count = get_last_frame_number(output_dir)

# # Video path (Change this for new videos)
# video_path = r"C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/Video_dataset/New folder/20250301_141434_2.mp4"
# cap = cv2.VideoCapture(video_path)

# # Get original video dimensions
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# print(f"Processing video: {video_path}")
# print(f"Starting from frame number: {frame_count}")
# print(f"Video Resolution: {width}x{height}")

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Process every second frame
#     if frame_count % 2 == 0:
#         resized_frame = cv2.resize(frame, (width, height))

#         # Save frame with continuous numbering
#         frame_filename = os.path.join(output_dir, f"frame_{frame_count:05}.jpg")
#         cv2.imwrite(frame_filename, resized_frame)

#     frame_count += 1

# cap.release()
# cv2.destroyAllWindows()

# print("Frame extraction completed.")







# import cv2
# import os

# # Define input video path
# video_path = r"C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/Video_dataset/New folder/20250301_141434_1.mp4"

# # Define output directory
# output_dir = r"C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/merge/"
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # Open the video
# cap = cv2.VideoCapture(video_path)

# # Get original video dimensions
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# print(f"Video Resolution: {width}x{height}")

# frame_count = 0

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Process every second frame (adjust as needed)
#     if frame_count % 2 == 0:
#         # Resize frame to match original video size
#         resized_frame = cv2.resize(frame, (width, height))

#         # Save the frame
#         frame_filename = os.path.join(output_dir, f"frame_{frame_count:04}.jpg")
#         cv2.imwrite(frame_filename, resized_frame)

#     frame_count += 1

# cap.release()
# cv2.destroyAllWindows()














"""no glob"""

# import cv2
# import os
# import glob
# from natsort import natsorted  # Import natural sorting

# # Path where extracted frames are stored
# frame_dir = r"C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/merge/"
# output_video = r"C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/Halol_raw1.mp4"

# # Get all frame files (sorted in natural order)
# frame_files = natsorted(glob.glob(os.path.join(frame_dir, "frame_*.jpg")))

# # Check if frames exist
# if not frame_files:
#     print("No frames found!")
#     exit()

# # Read the first frame to get video dimensions
# first_frame = cv2.imread(frame_files[0])
# if first_frame is None:
#     print("Error reading the first frame!")
#     exit()

# height, width, layers = first_frame.shape

# # Define the video writer (MP4 format, 30 FPS)
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
# fps = 30  # Change as needed
# out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# # Loop through and add frames to video
# for frame_file in frame_files:
#     frame = cv2.imread(frame_file)
#     if frame is None:
#         print(f"Error reading frame: {frame_file}")
#         continue
#     out.write(frame)

# # Release video writer
# out.release()
# print(f"Video saved as: {output_video}")
