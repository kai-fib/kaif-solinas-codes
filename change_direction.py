import keras_ocr
import cv2
#import math
import numpy as np
from os import listdir
import math
def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)
pipeline = keras_ocr.pipeline.Pipeline()
in_path = 'C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/Video_dataset/jalkal_07/exp/mid/'
out_path = 'C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/Video_dataset/jalkal_07/exp/mid_c/'
S = listdir(in_path)
for i in range(0,len(S)):
    img = keras_ocr.tools.read(in_path + S[i])
    image = np.copy(img)
    prediction_groups = pipeline.recognize([img])
    #Define the mask for inpainting
    mask = np.zeros(img.shape[:2], dtype="uint8")
    for j in range(0,len(prediction_groups[0])):
        QW = prediction_groups[0][j]
        if 'south' in QW[0]:
           #for box in QW[1]:  # for 'north'
               x0, y0 = QW[1][0]
               x1, y1 = QW[1][1]
               x2, y2 = QW[1][2]
               x3, y3 = QW[1][3]
               x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
               x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)
               thickness = int(math.sqrt( (x2 - x1)*2 + (y2 - y1)*2 ))
                       #Define the line and inpaint
               cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255,
                       thickness)
               image = cv2.inpaint(image, mask, 7, cv2.INPAINT_NS)
    cv2.putText(image, 'East', (int(x3), int(y3)), cv2.FONT_HERSHEY_SIMPLEX,0.65, (255, 255, 255), 2)
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path + S[i], bgr_image)