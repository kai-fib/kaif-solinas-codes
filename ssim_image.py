import cv2
from skimage.metrics import structural_similarity as ssim
import argparse

def calculate_ssim(imageA_path, imageB_path):
    # Read the images
    imageA = cv2.imread(imageA_path)
    imageB = cv2.imread(imageB_path)

    # Resize both images to the same size (optional, based on your use case)
    imageB = cv2.resize(imageB, (imageA.shape[1], imageA.shape[0]))

    # Convert to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # Compute SSIM
    score, diff = ssim(grayA, grayB, full=True)
    print(f"SSIM: {score:.4f}")


    # Optional: visualize the difference
    diff = (diff * 255).astype("uint8")
    cv2.imshow("Image A", imageA)
    cv2.imshow("Image B", imageB)
    cv2.imshow("Difference", diff)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    # Replace with your image paths
    img1_path = r"C:\Users\Kaif Ibrahim\Downloads\_(DE,3)_(NA)_(0-0-1)_(000.06M)_(L).png"
    img2_path = r"C:\Users\Kaif Ibrahim\Downloads\_(DE,3)_(NA)_(0-0-0)_(00.10M)_(L).png"
    calculate_ssim(img1_path, img2_path)

# import cv2

# def show_red_value(event, x, y, flags, param):
#     if event == cv2.EVENT_MOUSEMOVE:
#         # Get BGR value at (x, y)
#         b, g, r = img[y, x]
#         print(f"Position: ({x}, {y}) - Red Value: {r}")

# # Load the image
# img = cv2.imread(r"C:\Users\Kaif Ibrahim\Desktop\sample\003 - 03.13.41 PM (1).jpeg")  # Change to your image path

# if img is None:
#     print("Image not found!")
#     exit()

# # Create a window and set the mouse callback
# cv2.namedWindow("Image")
# cv2.setMouseCallback("Image", show_red_value)

# while True:
#     cv2.imshow("Image", img)
#     if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
#         break

# cv2.destroyAllWindows()
