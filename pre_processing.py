# In[1]
# '''
# # shadow removing

# from PIL import Image
# import re
# import cv2
# import numpy as np
# #import pytesseract
# #from pytesseract import Output
# from matplotlib import pyplot as plt
# import en_core_web_sm
# import json
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# img = cv2.imread("C:/Users/Kaif Ibrahim/Desktop/chettos/kaif.jpeg")
# def shadow_remove(img):
#     rgb_planes = cv2.split(img)
#     result_norm_planes = []
#     for plane in rgb_planes:
#         dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
#         bg_img = cv2.medianBlur(dilated_img, 21)
#         diff_img = 255 - cv2.absdiff(plane, bg_img)
#         norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
#         result_norm_planes.append(norm_img)
#     shadowremov = cv2.merge(result_norm_planes)
#     return shadowremov
# #Shadow removal
# shad = shadow_remove(img)
# cv2.imwrite('after_shadow_remove1.jpg', shad)

# # In[2]

# #converting video to frames
# import cv2
# #import numpy as np
# #from keras.models import load_model
# #from numpy import vstack
# #import os
# #from os import listdir
# #from os.path import isfile, join
# #from matplotlib import pyplot as plt


# def video_to_frames(outpath,test_video):
#     cap= cv2.VideoCapture(test_video)

#     i=1
#     while(cap.isOpened()):
#         ret, frame = cap.read()
#         if ret == False:
#             break
#         if i%5 == 0:
#             #j = i*2
#             filename = 'img_%06d.jpg'%i
#             #cv2.imwrite(str(i)+'.jpg',frame)
#             cv2.imwrite(outpath + filename,frame)
#         i+=1
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ ==   '__main__':
#   outpath = "C:/Users/User/Desktop/CMWSSB_kilpakam/New folder (2)/"
#   video_to_frames(outpath,'20241116_121021.mp4')
  
# # In[3]

# # method-I for image pre-processing
# #D:\P1_YOLO\pre_processing\Underwater-Image-Enhancement-based-on-Fusion-Python-main\Underwater-Image-Enhancement-based-on-Fusion-Python-main

# import FeatureWeight
# import ImageDecompose
# import SimplestColorBalance
# from FeatureWeight import LaplacianContrast, LocalContrast, Saliency, Exposedness
# from ImageDecompose import fuseTwoImages
# from SimplestColorBalance import simplest_cb
# import numpy as np
# import cv2
# import time
# from matplotlib import pyplot as plt
# import os
# from os import listdir

# s_path = "/content/drive/MyDrive/Colab_Notebooks/pre_process/video_frame_150322/"
# level = 5
# d_path = "/content/drive/MyDrive/Colab_Notebooks/pre_process/process_frame_150322_adv/"



# def applyCLAHE(img, L):
#     clahe = cv2.createCLAHE(clipLimit=2.0)
#     L2 = clahe.apply(L)
#     lab = cv2.split(img)
#     LabIm2 = cv2.merge((L2, lab[1], lab[2]))
#     img2 = cv2.cvtColor(LabIm2, cv2.COLOR_Lab2BGR)
#     result = []
#     result.append(img2)
#     result.append(L2)
#     return result

# def calWeight(img, L):
#     L = np.float32(np.divide(L, (255.0)))
#     WL = np.float32(LaplacianContrast(L)) # Check this line
#     WC = np.float32(LocalContrast(L))
#     WS = np.float32(Saliency(img))
#     WE = np.float32(Exposedness(L))
#     weight = WL.copy()
#     weight = np.add(weight, WC)
#     weight = np.add(weight, WS)
#     weight = np.add(weight, WE)
#     return weight

# S = listdir(s_path)

# for i in range(0,len(S)):
#     image = cv2.imread(s_path + S[i])

#     #image = cv2.imread('[000007].jpg')
#     #def enhance(image, level):
#     img1 = simplest_cb(image, 5)                    #color balancing
#     img1 = np.uint8(img1)
#     LabIm1 = cv2.cvtColor(img1, cv2.COLOR_BGR2Lab);
#     L1 = cv2.extractChannel(LabIm1, 0);
#     result = applyCLAHE(LabIm1, L1)
#     img2 = result[0]
#     L2 = result[1]
#     w1 = calWeight(img1, L1)
#     w2 = calWeight(img2, L2)
#     sumW = cv2.add(w1, w2)
#     w1 = cv2.divide(w1, sumW)
#     w2 = cv2.divide(w2, sumW)

#     final = fuseTwoImages(w1, img1, w2, img2, level)
#     fusion= np.uint8(final)
#     noiseless_image_colored = cv2.fastNlMeansDenoisingColored(fusion,None,15,15,7,21)
#     # cv2.imshow('Original_image',image)
#     # cv2.imshow('Processed_image.jpg',fusion)
#     #cv2.imshow('C.jpg',fusion)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     cv2.imwrite(d_path + S[i],noiseless_image_colored)
#     #fusion1 = cv2.cvtColor(fusion,cv2.COLOR_BGR2RGB)


# import cv2
# from os import listdir


# def convert_frames_to_video(pathIn,pathOut,fps):
#         #frame_array = []
#         #files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
#         S = listdir(pathIn)
#         out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
#         for i in range(0,len(S)):
#             #filename = pathIn + S[i]
#             img = cv2.imread(pathIn + S[i])
#             out.write(img)
#             #img1 = cv2.resize(img, [720,1280])
#             #frame_array.append(img)
#         out.release()


# pathIn= '/content/drive/MyDrive/Colab_Notebooks/pre_process/process_frame_150322_basic/'
# pathOut = '/content/drive/MyDrive/Colab_Notebooks/pre_process/Video_short1.mp4'
# fps = 25.0
# size = (1280,720)
# convert_frames_to_video(pathIn, pathOut, fps)
# '''
# In[4]

# method-II for image pre-processing
import cv2
import numpy as np
#from keras.models import load_model
from numpy import vstack
import os
from os import listdir
from os.path import isfile, join
#from matplotlib import pyplot as plt


def histo_eq_drak_bright(test_image):
    #input_img = cv2.imread('C:/Users/User/Desktop/test/test_images/wo_gan/New folder/New folder/img_v2_000194.jpg')
    img_yuv = cv2.cvtColor(test_image, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    # cv2.imshow('Histogram equalized', img_output)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #cv2.imwrite('histo_equ.jpg',img_output)
    return img_output

def image_enhan_deblurr(test_image):
    #input_img = cv2.imread(test_image)
    #image = cv2.cvtColor(input_img,cv2.COLOR_BGR2RGB)    ## for matplotlib only
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(test_image, -1, kernel)

    # cv2.imshow('A.jpg',sharpened_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #cv2.imwrite('image_enha.jpg',sharpened_image)

    return sharpened_image


#trained_weights = 'model_g_020000.h5'
s_path = 'C:/Users/Kaif Ibrahim/Desktop/chettos/'
d_path = 'C:/Users/Kaif Ibrahim/Desktop/task/'
S = listdir(s_path)
i=0

for i in range(0,len(S)):
    test_imag = cv2.imread(s_path + S[i])
    #test_imag = cv2.imread('0.jpg')
    #fin_img1 = turbid_water(trained_weights,test_imag)
    fin_img2 = histo_eq_drak_bright(test_imag)
    fin_img3 = image_enhan_deblurr(fin_img2)
    #j = 151
    cv2.imwrite(d_path + S[i],fin_img3)
    #cv2.imwrite(path+'image_gan_%02d.jpg'%j,fin_img3)
    #cv2.imwrite('image_pp2_26.jpg',fin_img3)
    # cv2.imshow('A.jpg', fin_img3)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# '''

# # In[5]

# # image denoising
# #https://www.projectpro.io/recipes/remove-noise-from-images-opencv
# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
# image = cv2.imread('/content/drive/MyDrive/Colab_Notebooks/pre_process/3.png',1)
# #image_bw = cv2.imread('/content/drive/MyDrive/Colab_Notebooks/pre_process/video_1/img_v1_000207.jpg',0)
# #noiseless_image_bw = cv2.fastNlMeansDenoising(image_bw, None, 20, 7, 21)
# noiseless_image_bw = cv2.fastNlMeansDenoisingColored(image,None,20,20,7,21)
# noiseless_image_colored = cv2.fastNlMeansDenoisingColored(image,None,15,15,7,21)
# titles = ['Original Image(colored)','Image after removing the noise (colored)', 'Original Image (grayscale)','Image after removing the noise (grayscale)']
# images = [image,noiseless_image_colored, noiseless_image_bw]
# plt.figure(figsize=(13,5))
# for i in range(4):
#     plt.subplot(2,2,i+1)
#     plt.imshow(cv2.cvtColor(images[i],cv2.COLOR_BGR2RGB))
#     plt.title(titles[i])
#     plt.xticks([])
#     plt.yticks([])
# plt.tight_layout()
# plt.show()


# # In[2]

# # image dehazing [11/07/2024]

# import image_dehazer
# import cv2
# from os import listdir

# s_path = 'C:/Users/User/Desktop/hoi/'
# d_path = 'C:/Users/User/Desktop/CMWSSB_kilpakam/final_images/'
# S = listdir(s_path)
# #i=0

# for i in range(0,len(S)):
#     HazeImg = cv2.imread(s_path + S[i])
#     HazeCorrectedImg, HazeTransmissionMap = image_dehazer.remove_haze(HazeImg)

#     cv2.imwrite(d_path + S[i],HazeCorrectedImg)
   
#     # cv2.imshow('A.jpg', fin_img3)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()

# # In[3]
    
# # converting numpy image to PIL image     
# from PIL import Image
# import numpy as np

# # Assuming you have your numpy array stored in np_arr
# new_im = Image.fromarray(np_arr)

# # To display the new image
# new_im.show()

# # In[4] 

# # brightness increase and decrease

# from PIL import Image, ImageEnhance

# img = Image.open("/content/drive/MyDrive/Colab_Notebooks/pre_process/img_v2_002036.jpg").convert("RGB")

# img_enhancer = ImageEnhance.Brightness(img)

# factor = 1
# enhanced_output = img_enhancer.enhance(factor)
# enhanced_output.save("/content/drive/MyDrive/Colab_Notebooks/pre_process/original-image.png")

# factor = 0.5
# enhanced_output = img_enhancer.enhance(factor)
# enhanced_output.save("/content/drive/MyDrive/Colab_Notebooks/pre_process/dark-image.png")

# factor = 1.5
# enhanced_output = img_enhancer.enhance(factor)
# enhanced_output.save("/content/drive/MyDrive/Colab_Notebooks/pre_process/bright-image.png")

# # In[4]

# import cv2
# from os import listdir


# def convert_frames_to_video(pathIn,pathOut,fps):
#         #frame_array = []
#         #files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
#         S = listdir(pathIn)
#         out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
#         for i in range(0,len(S)):
#             #filename = pathIn + S[i]
#             img = cv2.imread(pathIn + S[i])
#             out.write(img)
#             #img1 = cv2.resize(img, [720,1280])
#             #frame_array.append(img)
#         out.release()


# pathIn= 'C:/Users/User/Desktop/phonix_inspection/final/'
# pathOut = 'C:/Users/User/Desktop/phonix_inspection/Manual_processed1.mp4'
# fps = 25.0
# size = (1280,720)
# convert_frames_to_video(pathIn, pathOut, fps)

# # In[5]

# # reduce video size

# import moviepy.editor as mp

# clip = mp.VideoFileClip("merged.mp4")
# clip_resized = clip.resize(height=480)
# clip_resized.write_videofile("merged_resized.mp4")

# # In[6]

# from moviepy.editor import VideoFileClip  

# # Load the AVI file  
# video = VideoFileClip("20231027_122132_1.avi")  

# # Write it to an MP4 file  
# video.write_videofile("20231027_122132_1_output.mp4")
# '''