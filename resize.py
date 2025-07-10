import cv2
img = cv2.imread("C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/Video_dataset/background_images/misclassification/select/deform/frame_1420.jpg")
k=cv2.resize(img,(1280,720))
cv2.imwrite("C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/Video_dataset/background_images/misclassification/select/deform/k.jpg",k)