'''
https://blog.roboflow.com/computer-vision-measure-distance/
https://pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/
https://www.cmrp.com/ovalitycalc-php-template
relation between numpy and opencv
1. x and y coordinates are interchange
2. rgb channels are interchange

Version-3 combination of first and second model

'''
# video to frame
# import cv2
# import os

# output_dir = "D:/New folder/frames/"
# os.makedirs(output_dir, exist_ok=True)

# vid = cv2.VideoCapture("D:/New folder/23_aug_3_crop.avi")

# success, frame = vid.read()
# count = 0

# while success:
#     frame_filename = os.path.join(output_dir, f"frame_{count:05d}.jpg")
#     cv2.imwrite(frame_filename, frame)

#     success, frame = vid.read()
#     count += 1

# vid.release()
# ###############################################################################
"""Frames to Video"""
# import cv2
# import os

# # Path where frames are stored
# input_dir = "D:/results/23_aug/1/new_rgb/"
# output_video = "D:/results/23_aug/1/23_aug_1_laser_dia.mp4"

# # Get list of frames sorted by name
# frames = sorted([f for f in os.listdir(input_dir) if f.endswith(".jpg")])

# # Read the first frame to get height and width
# first_frame = cv2.imread(os.path.join(input_dir, frames[0]))
# height, width, layers = first_frame.shape

# # Define codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
# fps = 24  # Adjust FPS as per original video
# out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# # Write frames into video
# for frame_name in frames:
#     frame_path = os.path.join(input_dir, frame_name)
#     img = cv2.imread(frame_path)
#     out.write(img)

# out.release()
# print(f"MP4 video saved at {output_video}")

# ###############################################################################
# # import the opencv library
# # for live video 
# import cv2 


# # define a video capture object 
# vid = cv2.VideoCapture(0) 

# while(True): 
	
# 	# Capture the video frame 
# 	# by frame 
# 	ret, frame = vid.read() 

# 	# Display the resulting frame 
# 	cv2.imshow('frame', frame) 
	
# 	# the 'q' button is set as the 
# 	# quitting button you may use any 
# 	# desired button of your choice 
# 	if cv2.waitKey(1) & 0xFF == ord('q'): 
# 		break

# # After the loop release the cap object 
# vid.release() 
# # Destroy all the windows 
# cv2.destroyAllWindows() 
    
# ###############################################################################

# # design the cheker board
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt

# black = np.zeros([200,200],dtype = 'uint8')
# white = (np.ones([200,200],dtype = 'uint8'))*255

# #[rows, col] = A.shape

# k = 0
# k1 = 0
# r = 1200
# merged_img = []
# merged_img1 = []

# # odd rows
# for i in range(0,r,200):
#     if k%2 == 0:
#        G = white
#     else:
#        G = black
#     merged_img.append(G)
#     k = k+1

# result_h = np.hstack(merged_img)

# # even rows
# for i1 in range(0,r,200):
#     if k1%2 == 0:
#        G1 = black
#     else:
#        G1 = white
#     merged_img1.append(G1)
#     k1 = k1+1

# result_h1 = np.hstack(merged_img1)

# verticall = []
# for u in range (0,5):
#     if u%2 == 0:
#         F = result_h1
#     else:
#         F = result_h
#     verticall.append(F)


# result_v = np.vstack(verticall)



# #cv2.imshow('A.jpg',result_h)
# #cv2.imshow('A1.jpg',result_h1)
# cv2.imshow('A1.jpg',result_v)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imwrite("p_checkerboad3.png",result_v)   

# ###############################################################################
# ## detecting the checkerboard edge

# import cv2

# # Load the image
# img = cv2.imread('F:/23.02.2024/240223-002/14.jpeg')
# #img = cv2.GaussianBlur(org_img, (25,25), 21)
# #img = cv2.resize(img1,[1024,720])

# # Define the number of rows and columns in the chessboard
# n_rows = 6
# n_cols = 8

# # Convert the image to grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Find the corners of the chessboard
# ret, corners = cv2.findChessboardCorners(gray, (n_rows, n_cols), None)

# # Refine the corners to subpixel accuracy
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# corners = cv2.cornerSubPix(gray, corners, (21, 21), (-1, -1), criteria)

# # Draw the corners on the image
# cv2.drawChessboardCorners(img, (n_rows, n_cols), corners, ret)

# gh = corners.reshape(len(corners),2)

# # Display the image
# cv2.imshow('Image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
 
# ###############################################################################

# # import cv2
# # import numpy as np
# # import os
# # import glob
# # from scipy.spatial import distance as dist 
# # # Defining the dimensions of checkerboard
# # CHECKERBOARD = (6,8)
# # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# # # Creating vector to store vectors of 3D points for each checkerboard image
# # objpoints = []
# # # Creating vector to store vectors of 2D points for each checkerboard image
# # imgpoints = [] 
# # focal = []
# # #distance = 500
# # actual_length = 26
# # # Defining the world coordinates for 3D points
# # objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
# # objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
# # #objp = objp * 26
# # #prev_img_shape = None

# # # Extracting path of individual image stored in a given directory
# # images = glob.glob('D:/laser profiling/240222-003_mds_testing/New folder/*.jpeg')
# # for fname in images:
# #     #img = cv2.imread(fname)
# #     img = cv2.imread('D:/laser profiling/240222-003_mds_testing/New folder/3.jpeg')
# #     #img = cv2.resize(img,[1280,720],interpolation = cv2.INTER_CUBIC)
# #     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# #     # Find the chess board corners
# #     # If desired number of corners are found in the image then ret = true
# #     ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+
# #     	cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    
# #     """
# #     If desired number of corner are detected,
# #     we refine the pixel coordinates and display 
# #     them on the images of checker board
# #     """
# #     if ret == True:
# #         objpoints.append(objp)
# #         # refining pixel coordinates for given 2d points.
# #         corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        
# #         imgpoints.append(corners2)

# #         # Draw and display the corners
# #         img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2,ret)
    
# #     cv2.imshow('img',img)
# #     cv2.waitKey(0)

# #     NCC1 = np.reshape(corners,[48,2])

# #     NC1 = NCC1.copy()
  
# #     NC1[:,0] = NCC1[:,1]
# #     NC1[:,1] = NCC1[:,0]
# #     h,w = img.shape[:2]
    
    
# #     d1 = NC1[1,0]-NC1[0,0]
# #     d2 = NC1[2,0]-NC1[1,0]
# #     d3 = NC1[3,0]-NC1[2,0]
# #     d4 = NC1[4,0]-NC1[3,0]
# #     d5 = NC1[5,0]-NC1[4,0]
    
# #     d = (d1+d2+d3+d4+d5)/5

# #     d05 = NC1[5,0]-NC1[0,0] 
    
# # cv2.destroyAllWindows()




# ###############################################################################

# https://stackoverflow.com/questions/62698756/opencv-calculating-orientation-angle-of-major-and-minor-axis-of-ellipse
#start 
""" laser bottom missing"""
import cv2
import numpy as np
from scipy.spatial import distance as dist 
from os import listdir 
from matplotlib import pyplot as plt
import circle_fit as cf
import math

# input_path = "D:/New folder/frames/"
# #input_path = 'D:/laser profiling/Result_26122023/case4/original_image_296mm/'
# output_path_rgb = "D:/New folder/new_rgb/"
# output_path_bw = "D:/New folder/new_bw/"


# diameter_known = 1

# img_list = listdir(input_path)
# PPM = None

# orig_dia_mm = 1200

# R_pix =[] # radius in pixels
# D_mm = []
# Ovality_all = []
# Csl_rad = []
# DIA_MET = []


# #MMP = 26/53.2 ## first col average, 26 mm is actual length of checkerbox- img3
# #MMP = (26/53.2628) # first block - img3
# #MMP = (26/53.299)



# #ind = 1
# for ind in range(0,len(img_list)): 
    
#     combine_rad = []
#     sub_rad_dis = []
#     mul_dia_mm  = []

#     mul_rad_pix = []
#     multiple_xc = []
#     multiple_yc = []
    
#     # img = cv2.imread('D:/laser_image/laser.png')
#     img = cv2.imread(input_path + img_list[ind])

#     [row, column] = img.shape[:2]
    
    
#     blue,green,red = cv2.split(img)
#     new_red   = np.zeros([row,column],dtype = 'uint8')
#     # new_image = (np.ones([row,column,3],dtype = 'uint8'))*255
#     # new_green = np.zeros([row,column],dtype = 'uint8')
#     # new_blue  = np.zeros([row,column],dtype = 'uint8')
#     # color based thresholding
#     for ia in range(0,row):
#         for ja in range(0,column):
#             rp = red[ia,ja]
#             gp = green[ia,ja]
#             bp = blue[ia,ja]
            
#             if (rp>=30 and gp<=100 and bp<=100):  # for embedded laser circle
#                 pixel_r = 0
#                 pixel_g = 0
#                 pixel_b = 0
#             elif (rp>=130 and gp>=130 and bp>=130):  #for white text
#                 pixel_r = 0
#                 pixel_g = 0
#                 pixel_b = 0
#             elif ((rp>175) and (gp<255 and bp<255)): 
#                 pixel_r = 255
#                 pixel_g = 0
#                 pixel_b = 0
#             else:
#                 pixel_r = 0
#                 pixel_g = 0
#                 pixel_b = 0
#             new_red[ia,ja]= pixel_r
#             removal_y_threshold = int(row * 0.68) #put
#             new_red[removal_y_threshold:, :] = 0
#             # new_green[i,j]= pixel_g
#             # new_blue[i,j]= pixel_b


#     # cv2.imshow('a.jpg',img)
#     # cv2.imshow('b.jpg',new_red)
#     # cv2.waitKey()
#     # cv2.destroyAllWindows()
    
#     #newRGBImage = cv2.merge((new_blue,new_green,new_red))
    
#     #yx_coords1 = np.column_stack(np.where(new_red == 255))
#     # xy_coords = yx_coords.copy()

#     # xy_coords[:,0] = yx_coords[:,1]
#     # xy_coords[:,1] = yx_coords[:,0]
    
#     xy_coords2 = cv2.findNonZero(new_red)    # finding the edge pixels coordinates
#     xy_coords  = xy_coords2.reshape(len(xy_coords2),2)

#     # for (x, y) in xy_coords:                               # green dots on original image
#     #     cv2.circle(img, (x, y), 1, (0, 255, 0), -1)        # green dots on original image
#     #     cv2.circle(new_red, (x, y), 1, (255, 255, 255), -1) # white dots on bw mask

#         # removal_y_threshold = int(row * 0.65) 
#         # new_red[removal_y_threshold:, :] = 0

#     # Show images with dots
#     # cv2.imshow('a.jpg', img)
#     # cv2.imshow('b.jpg', new_red)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
    
#     '''
#     A = np.array([1,0,5,8,0,0,4,7])
#     B = cv2.findNonZero(A)
#     bb = len(B)
#     C = B.reshape(bb,2)
#     '''
#     xc,yc,r,_ = cf.least_squares_circle(xy_coords)
   
#     # # DIAGONAL FEATURE ADDITION END
#     # for ib in range(0,len(xy_coords)):
#     #     sb = xy_coords[ib]
#     #     calcu_rad = int(dist.euclidean((xc,yc),(sb[0],sb[1])))
#     #     combine_rad.append(calcu_rad)
#         #cv2.circle(img,(sb[0],sb[1]),2,(100,255,150),-1) # original laser circle coordinates
        
#     #R.append(r) # radius in pixels
#     if diameter_known == 1:
        
#         if PPM == None:
#             PPM = (2*r)/(orig_dia_mm)
#             Rad = (2*r/PPM)
          
#         else:
#             Rad = ((2*r)/PPM) 
          
        
#         #R_pix.append(r) # radius in pixels
#         D_mm.append(Rad)
    
#     else:
        
#         Rad = MMP *2*r
#         #Csl_rad.append(Rad)
    
    
#     cv2.circle(img,(int(xc),int(yc)),int(r),(255,0,0),2) # fit circle
#     #cv2.circle(img,(int(xc),int(yc)),5,(0,0,255),-1)  # fit circle radius
#     #cv2.circle(img,(int(column/2),int(row/2)),5,(0,255,255),-1)  # fit circle radius
    
#     # cv2.circle(new_red,(int(xc),int(yc)),int(r),(255,0,0),2) # fit circle
#     # cv2.circle(img,(int(xc),int(yc)),5,(0,0,255),-1)  # fit circle radius
#     #cv2.circle(new_red,(int(column/2),int(row/2)),5,(0,255,255),-1)  # fit circle radius
#     Angle = []  
#     pos_Angle = []
#     for ib in range(0,len(xy_coords)):
#         sb = xy_coords[ib]
#         calcu_rad = int(dist.euclidean((xc,yc),(sb[0],sb[1])))
#         combine_rad.append(calcu_rad)
        

#         #cv2.circle(img,(sb[0],sb[1]),2,(100,255,150),-1) # original laser circle coordinates
#         #cv2.line(img, (int(xc),int(yc)) ,(sb[0],sb[1]), (0,0,255), 2) #line connecting center and circle edges
        
        
#         angle_radians = math.atan2(((sb[1]-yc)), ((sb[0]-xc)))

# # Convert the angle to degrees
#         angle_degrees = math.degrees(angle_radians)
        
#         Angle.append(int(angle_degrees)) 
#         if angle_degrees < 0 :
#             angle_deg_pos = 360+ angle_degrees
#         else:
#             angle_deg_pos = angle_degrees
           
#         pos_Angle.append(int(angle_deg_pos)) 

        
#         angle_sorting = np.sort(Angle)
#         angle_int = angle_sorting.astype(int)
        
#     XY_COR = np.array(xy_coords)
#     COMBINE_RAD = np.array(combine_rad)
#     ANGLE = np.array(Angle)
#     pos_ANGLE = np.array(pos_Angle)
#     combine_mat = np.column_stack((pos_ANGLE, COMBINE_RAD, XY_COR,ANGLE))
#     sort_combine_mat = combine_mat[combine_mat[:,0].argsort()]   
#     del sb
#     rad_23_67 = []
#     rad_67_112 = []
#     rad_112_157 = []
#     rad_157_202 = []
#     rad_202_247 = []
#     rad_247_292 = []
#     rad_292_337 = []
#     rad_337_23 = []
#     for ic in range(0,len(xy_coords)):
#         an = combine_mat[ic,0]
#         sbx = combine_mat[ic,2]
#         sby = combine_mat[ic,3]
#         # if an < 25 or an > 340: # show all the negative angle
#         #     sbx = combine_mat[ic,2]
#         #     sby = combine_mat[ic,3]
#         #     cv2.circle(img,(sbx,sby),2,(100,255,150),-1)
        
#         # if 23 < an < 67 : # show all the negative angle
#         #     #sbx = combine_mat[ic,2]
#         #     #sby = combine_mat[ic,3]
            
#         #     rad_23_67.append(combine_mat[ic,1])
#         #     #cv2.circle(img,(sbx,sby),1,(0,0,0),-1)
#         #     #cv2.line(img, (int(xc),int(yc)) ,(sbx,sby), (0,0,0), 2)
        
#         # if 67 < an < 112 : # show all the negative angle
#         #     #sbx = combine_mat[ic,2]
#         #     #sby = combine_mat[ic,3]
#         #     rad_67_112.append(combine_mat[ic,1])
#         #    # cv2.circle(img,(sbx,sby),1,(0,0,0),-1)
            
#         # elif 112 < an < 157: # show all the negative angle
#         #     #sbx = combine_mat[ic,2]
#         #     #sby = combine_mat[ic,3]
#         #     rad_112_157.append(combine_mat[ic,1])
#         #     #cv2.circle(img,(sbx,sby),1,(0,0,0),-1)
            
#         if 157 < an < 202: # show all the negative angle
#             #sbx = combine_mat[ic,2]
#             #sby = combine_mat[ic,3]
#             rad_157_202.append(combine_mat[ic,1])
#             #cv2.circle(img,(sbx,sby),1,(0,0,0),-1)
            
#         # elif 202 < an < 247: # show all the negative angle
#         #     #sbx = combine_mat[ic,2]
#         #     #sby = combine_mat[ic,3]
#         #     rad_202_247.append(combine_mat[ic,1])
#         #     #cv2.circle(img,(sbx,sby),1,(0,0,0),-1)
            
#         # elif 247 < an < 292: # show all the negative angle
#         #     #sbx = combine_mat[ic,2]
#         #     #sby = combine_mat[ic,3]
#         #     rad_247_292.append(combine_mat[ic,1])
#         #     #cv2.circle(img,(sbx,sby),1,(0,0,0),-1)
            
#         # elif 292 < an < 337: # show all the negative angle
#         #     #sbx = combine_mat[ic,2]
#         #     #sby = combine_mat[ic,3]
#         #     rad_292_337.append(combine_mat[ic,1])
#         #     #cv2.circle(img,(sbx,sby),1,(0,0,0),-1)
            
#         elif an < 23 or an > 337: # show all the negative angle
#             #sbx = combine_mat[ic,2]
#             #sby = combine_mat[ic,3]
#             rad_337_23.append(combine_mat[ic,1])
#             #cv2.circle(img,(sbx,sby),1,(0,0,0),-1)
    
#     # rad_23 = np.mean(rad_23_67)
#     # rad_67 = np.mean(rad_67_112)
#     # rad_112 = np.mean(rad_112_157)
#     rad_157 = np.mean(rad_157_202)
#     # rad_202 = np.mean(rad_202_247)
#     # rad_247 = np.mean(rad_247_292)
#     # rad_292 = np.mean(rad_292_337)
#     rad_337 = np.mean(rad_337_23)
        
#     # end_point_23_67 = (
#     #         int(xc + rad_23 * math.cos(math.radians(45))),
#     #         int(yc - rad_23 * math.sin(math.radians(45)))
#     #     ) 
#     # cv2.line(img, (int(xc),int(yc)), end_point_23_67 , (255,211,0), 2)
#     # dia_23 = ((2* rad_23)/PPM)
#     # cv2.putText(img, "Dia 4 = {:.1f}mm".format(dia_23),(end_point_23_67[0],end_point_23_67[1]), cv2.FONT_HERSHEY_SIMPLEX,0.55, (0, 0, 0), 2)
#     # cv2.line(new_image, (int(xc),int(yc)), end_point_23_67 , (255,0,0), 1)
#     # cv2.putText(new_image, "Dia 4",(end_point_23_67[0],end_point_23_67[1]), cv2.FONT_HERSHEY_SIMPLEX,0.55, (0, 0, 0), 2)
#     # cv2.circle(new_image,(end_point_23_67[0],end_point_23_67[1]),1,(0,0,0),-1)


    
#     # end_point_67_112 = (
#     #         int(xc + rad_67 * math.cos(math.radians(90))),
#     #         int(yc - rad_67 * math.sin(math.radians(90)))
#     #     ) 
#     # cv2.line(img, (int(xc),int(yc)), end_point_67_112 , (255,0,0),1)
#     # dia_67 = ((2* rad_67)/PPM)
#     # cv2.putText(img, "Dia 3 = {:.1f}mm".format(dia_67),(end_point_67_112[0],end_point_67_112[1]), cv2.FONT_HERSHEY_SIMPLEX,0.55, (0, 0, 0), 2)
    
#     # # cv2.line(new_image, (int(xc),int(yc)), end_point_67_112 , (255,0,0), 1)
#     # # cv2.putText(new_image, "Dia 3",(end_point_67_112[0],end_point_67_112[1]-10), cv2.FONT_HERSHEY_SIMPLEX,0.55, (0, 0, 0), 2)
#     # # cv2.circle(new_image,(end_point_67_112[0],end_point_67_112[1]),1,(0,0,0),-1)

    
#     # end_point_112_157 = (
#     #         int(xc + rad_112 * math.cos(math.radians(135))),
#     #         int(yc - rad_112 * math.sin(math.radians(135)))
#     #     ) 
#     # cv2.line(img, (int(xc),int(yc)), end_point_112_157 , (255,0,100), 2)
#     # dia_112 = ((2* rad_112)/PPM)
#     # cv2.putText(img, "Dia 2 = {:.1f}mm".format(dia_112),(end_point_112_157[0],end_point_112_157[1]), cv2.FONT_HERSHEY_SIMPLEX,0.55, (0, 0, 0), 2)
    
#     # cv2.line(new_image, (int(xc),int(yc)), end_point_112_157 , (255,0,0), 1)
   
#     # cv2.putText(new_image, "Dia 2",(end_point_112_157[0],end_point_112_157[1]), cv2.FONT_HERSHEY_SIMPLEX,0.55, (0, 0, 0), 2)
#     # cv2.circle(new_image,(end_point_112_157[0],end_point_112_157[1]),1,(0,0,0),-1) 
    
    
#     end_point_157_202 = (
#             int(xc + rad_157 * math.cos(math.radians(180))),
#             int(yc - rad_157 * math.sin(math.radians(180)))
#         ) 
#     cv2.line(img, (int(xc),int(yc)), end_point_157_202 , (0,250,0), 2)
#     dia_157 = ((2* rad_157)/PPM)
#     cv2.putText(img, "Dia = {:.1f}mm".format(Rad),(end_point_157_202[0]+10,end_point_157_202[1]-20), cv2.FONT_HERSHEY_SIMPLEX,0.70, (0, 0, 0), 3)

#     # cv2.line(new_image, (int(xc),int(yc)), end_point_157_202 , (255,0,0), 1)
    
#     # cv2.putText(new_image, "Dia 1",(end_point_157_202[0]+10,end_point_157_202[1]), cv2.FONT_HERSHEY_SIMPLEX,0.55, (0, 0, 0), 2)
#     # cv2.circle(new_image,(end_point_157_202[0],end_point_157_202[1]),1,(0,0,0),-1) 

    
#     # end_point_202_247 = (
#     #         int(xc + rad_202 * math.cos(math.radians(225))),
#     #         int(yc - rad_202 * math.sin(math.radians(225)))
#     #     ) 
#     # cv2.line(img, (int(xc),int(yc)), end_point_202_247 , (255,232,0), 2)
#     # # cv2.line(new_image, (int(xc),int(yc)), end_point_202_247 , (255,0,0), 1)
#     # # cv2.circle(new_image,(end_point_202_247[0],end_point_202_247[1]),1,(0,0,0),-1) 

    
    
#     # end_point_247_292 = (
#     #         int(xc + rad_247 * math.cos(math.radians(270))),
#     #         int(yc - rad_247 * math.sin(math.radians(270)))
#     #     ) 
#     # cv2.line(img, (int(xc),int(yc)), end_point_247_292 , (255,0,215), 2)
#     # # cv2.line(new_image, (int(xc),int(yc)), end_point_247_292 , (255,0,0), 1)
#     # # cv2.circle(new_image,(end_point_247_292[0],end_point_247_292[1]),1,(0,0,0),-1) 
    

    
#     # end_point_292_337 = (
#     #         int(xc + rad_292 * math.cos(math.radians(315))),
#     #         int(yc - rad_292 * math.sin(math.radians(315)))
#     #     ) 
#     # cv2.line(img, (int(xc),int(yc)), end_point_292_337 , (255,234,0), 2)
#     # # cv2.line(new_image, (int(xc),int(yc)), end_point_292_337 , (255,0,0), 1)
#     # # cv2.circle(new_image,(end_point_292_337[0],end_point_292_337[1]),1,(0,0,0),-1) 
    
#     end_point_337_23 = (
#             int(xc + rad_337 * math.cos(math.radians(0))),
#             int(yc - rad_337 * math.sin(math.radians(0)))
#         ) 
#     cv2.line(img, (int(xc),int(yc)), end_point_337_23 , (0,250,0), 2)
#     # cv2.line(new_image, (int(xc),int(yc)), end_point_337_23 , (255,0,0), 1)
#     # cv2.circle(new_image,(end_point_337_23[0],end_point_337_23[1]),1,(0,0,0),-1) 

        
    
    
#         # if an < 24:
#         #     an_n = 180 + an
#         #     new_Angle.append(an_n)
#         # ANGLE_n = np.array(new_Angle)
#         # combine_mat_n = np.column_stack((ANGLE_n, COMBINE_RAD, XY_COR))
#         # sort_combine_mat_n = combine_mat_n[combine_mat_n[:,0].argsort()]     
#     '''
#         sorting a matrix based on the first column value
#         import numpy as np

#         A = np.array([0,5,9,8,6])
#         B = np.array([1,8,6,9,7])
#         C = np.array([7,3,6,5,4])

#         D = np.column_stack((A,B,C))


#         E = D[D[:, 0].argsort()]
#     '''
        
#     # final_avg_dia =  (dia_23 + dia_112 )/2  
#     #DIA_MET.append(final_avg_dia)
#     # cv2.putText(img, "Dia = {:.1f}mm".format(final_avg_dia),(100,480), cv2.FONT_HERSHEY_SIMPLEX,0.65, (0, 0, 0), 2)
#     # cv2.putText(new_image, "Dia 1 = {:.1f}mm".format(dia_157),(100,450), cv2.FONT_HERSHEY_SIMPLEX,0.65, (0, 0, 0), 2)
#     # cv2.putText(new_image, "Dia 2 = {:.1f}mm".format(dia_112),(100,480), cv2.FONT_HERSHEY_SIMPLEX,0.65, (0, 0, 0), 2)
#     # cv2.putText(new_image, "Dia 3 = {:.1f}mm".format(dia_67),(100,510), cv2.FONT_HERSHEY_SIMPLEX,0.65, (0, 0, 0), 2)
#     # cv2.putText(new_image, "Dia 4 = {:.1f}mm".format(dia_23),(100,540), cv2.FONT_HERSHEY_SIMPLEX,0.65, (0, 0, 0), 2)
    
#     # cv2.putText(new_image, "Avg Dia = {:.1f}mm".format(final_avg_dia),(100,580), cv2.FONT_HERSHEY_SIMPLEX,0.65, (0, 0, 0), 2)
#     # ovality calculation
    
#     CR = np.array(combine_rad) 
#     all_measure_radius = np.sort(CR)
    
#     # calculate distance between center and the each edge point; if the distance is more than or less than 10 then remove those points 
#     index_end = np.where(all_measure_radius>((r+1))) 
#     index_start = np.where(all_measure_radius<(r-1))
    
#     new_measure_radius = np.delete(all_measure_radius, index_end)  
#     new_measure_radius = np.delete(new_measure_radius, index_start) # deleting all calculated radius value which is out of index 
#     '''
#     A = np.array([1,2,5,9,8,3,7])
    
#     B = np.sort(A)
    
#     index_end = np.where(B>8)
#     index_start = np.where(B<2)
    
#     C = B.copy()
    
#     new_B = np.delete(B, index_end)
#     new_B = np.delete(new_B, index_start) 
#     '''
#     ovality_cal = ((new_measure_radius[-1] - new_measure_radius[0]) /r)*100
#     Ovality_all.append(ovality_cal)
#     cv2.putText(img, "Diameter: {:.1f}mm".format(Rad),(20,300), cv2.FONT_HERSHEY_SIMPLEX,0.85, (0, 0, 255), 4)
#     cv2.putText(img, "Ovality: {:.1f} %".format(ovality_cal),(20,350), cv2.FONT_HERSHEY_SIMPLEX,0.85, (0, 100, 255), 4) #make 400
    
#     cv2.putText(new_red, "Diameter: {:.1f}mm".format(Rad),(20,300), cv2.FONT_HERSHEY_SIMPLEX,0.85, (255, 255, 255), 4)
#     cv2.putText(new_red, "Ovality: {:.1f} %".format(ovality_cal),(20,350), cv2.FONT_HERSHEY_SIMPLEX,0.85, (255, 255, 255), 4) #make 400
    
#     cv2.imwrite(output_path_rgb + img_list[ind],img)
#     cv2.imwrite(output_path_bw + img_list[ind],new_red)
    
    # cv2.imshow('a1.jpg',img)
    # cv2.imshow('b1.jpg',new_red)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    


# Cslrad = np.array(Csl_rad)    

# np.save('Calculate_dia.npy',Cslrad)
# length = len(Cslrad)
# call_diaa = np.mean(Cslrad) 
# q_101 = np.repeat(call_diaa,length)  




#np.save('pixel_radius.npy',R_pix)
# np.save('D:/results/22_aug/7/Dia_mm.npy',D_mm)
# np.save('D:/results/22_aug/7/ovality_percentage.npy',Ovality_all)   

D_mm = np.load("D:/Dia_mm.npy/") # for loading
Ovality_all = np.load('D:/results/23_aug/3/ovality_percentage.npy') #for loading

Mean_dia = np.mean(D_mm)
Oval1 = np.array(Ovality_all)   
#R1 = np.array(R_pix)
D1 = np.array(D_mm)
#mean_rad_pix = np.mean(D1)
length = len(D1)
q_100 = list(range(0,length))
#mean_cal_diameter = np.repeat(Mean_dia,length)
mean_cal_diameter = np.repeat(152.4,length)
#mean_cal_rad = np.repeat(mean_rad_pix,length)  

#dia 


fig = plt.figure(figsize=(10,5))
# plt.ylim(1000,1500)
# plt.xlim(0,300) #21_8 aug
plt.plot(q_100, D1,'black')
plt.title("Diameter Measure")
plt.legend(['Diameter mm'])
# plt.plot(x_values, dia_data,'black')
plt.xlabel('pipe length')
plt.ylabel('Diameter')
fig.savefig('D:/results/23_aug/1/dia_pix1.jpg', bbox_inches='tight', dpi=150)
plt.show()



# for ovality

# fig = plt.figure(figsize=(10,5))
# # plt.ylim(0.3100,0.5200)
# # plt.xlim(0,300)
# plt.plot(q_100, Oval1,'black')
# plt.title("Ovality Measure")
# plt.legend(['Ovality'])
# plt.xlabel('pipe length')
# plt.ylabel('Diameter')
# fig.savefig('D:/New folder/new_rgb/oval_pix1.jpg', bbox_inches='tight', dpi=150)
# plt.show()  
