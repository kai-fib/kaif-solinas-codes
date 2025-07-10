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

# vid = cv2.VideoCapture('Pipe2--20230526_132936_1.avi')
# sucess,frame = vid.read()
# count = 0


# while sucess:
#     count+=1
#     suce, fname = vid.read()
#     if suce == False:
#         break
#     cv2.imwrite('image_%05d.jpg' % count,fname)
    
# ###############################################################################
# import cv2
 
# # Opens the Video file
# cap= cv2.VideoCapture('20230926_133349_1 - Trim - Trim.avi')
# i=1
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret == False:
#         break
#     if i%2 == 0:
#         filename = 'img_%05d.jpg'%i
#         #cv2.imwrite('kang'+str(i)+'.jpg',frame)
#         cv2.imwrite(filename,frame)
#     i+=1
 
# cap.release()
# cv2.destroyAllWindows() 
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
    
###############################################################################

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

###############################################################################
## detecting the checkerboard edge

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
 




###############################################################################

# https://stackoverflow.com/questions/62698756/opencv-calculating-orientation-angle-of-major-and-minor-axis-of-ellipse
#start 

import cv2
import numpy as np
from scipy.spatial import distance as dist 
from os import listdir 
from matplotlib import pyplot as plt
import circle_fit as cf
import math

#import math

input_path = 'C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/najiya/original_images/'
#D:/P2_laser profiling/2902_mds_final_unknown_dia/small_wheel/mds_390/
output_path_rgb = 'C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/najiya/output_rgb1/'
output_path_format = 'C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/najiya/output_sample1/'
#output_path_bw = 'C:/Users/User/Desktop/laser_field/20240529_215416_1_fold/processed_bw/'


diameter_known = 1

img_list = listdir(input_path)
PPM = None
#PPM = 1.7623276664724357
orig_dia_mm = 390

R_pix =[] # radius in pixels
D_mm = []
Ovality_all = []
Csl_rad = []
DIA_MET = []

#MMP = 26/53.2 ## first col average, 26 mm is actual length of checkerbox- img3
#MMP = (26/53.2628) # first block - img3
MMP = (26/53.299)



ind = 100
for ind in range(0,len(img_list)):
    
    combine_rad = []
    sub_rad_dis = []
    mul_dia_mm  = []

    mul_rad_pix = []
    multiple_xc = []
    multiple_yc = []
    
    #img = cv2.imread('C:/Users/User/Desktop/laser_field/20240529_213720_1_fold/original_image/img_03336.jpg')
    img = cv2.imread(input_path + img_list[ind])

    [row, column] = img.shape[:2]
    
    
    blue,green,red = cv2.split(img)
    new_red   = np.zeros([row,column],dtype = 'uint8')
    new_image = (np.ones([row,column,3],dtype = 'uint8'))*255
    # new_blue  = np.zeros([row,column],dtype = 'uint8')
    # color based thresholding
    for ia in range(0,row):
        for ja in range(0,column):
            rp = red[ia,ja]
            gp = green[ia,ja]
            bp = blue[ia,ja]
            
            if (rp>200 and gp>200 and bp>200):
                pixel_r = 0 
                pixel_g = 0
                pixel_b = 0
            elif ((rp>249) and (gp<255 and bp<255)):
                pixel_r = 255 
                pixel_g = 0
                pixel_b = 0
            else:
                pixel_r = 0 
                pixel_g = 0
                pixel_b = 0
           
            
            new_red[ia,ja]= pixel_r
            # new_green[i,j]= pixel_g
            # new_blue[i,j]= pixel_b

    k=cv2.imread('a.jpg',img)
    n=cv2.imread('b.jpg',new_red)
    plt.imshow(k)
    plt.imshow(n)
    #cv2.waitKey()
    #cv2.destroyAllWindows()
    #
    #newRGBImage = cv2.merge((new_blue,new_green,new_red))
    
    #yx_coords1 = np.column_stack(np.where(new_red == 255))
    # xy_coords = yx_coords.copy()

    # xy_coords[:,0] = yx_coords[:,1]
    # xy_coords[:,1] = yx_coords[:,0]
    
    xy_coords2 = cv2.findNonZero(new_red)    # finding the edge pixels coordinates
    xy_coords  = xy_coords2.reshape(len(xy_coords2),2)
    
    '''
    A = np.array([1,0,5,8,0,0,4,7])
    B = cv2.findNonZero(A)
    bb = len(B)
    C = B.reshape(bb,2)
    '''
    xc,yc,r,_ = cf.least_squares_circle(xy_coords)
    print(f'this is xc from cf: {xc}')
    print(f'this is yc from cf: {yc}')
    print(f'this is r from cf: {r}')
    print(f'this is the whole cf : {cf.least_squares_circle(xy_coords)}')
    
    #R.append(r) # radius in pixels
    if diameter_known == 1:
        
        if PPM == None:
            PPM = (2*r)/(orig_dia_mm)
            Rad = (2*r/PPM)
          
        else:
            Rad = ((2*r)/PPM) 
          
        
        # #R_pix.append(r) # radius in pixels
        # if Rad > (orig_dia_mm + 5):
        #     Rad = orig_dia_mm
        
        D_mm.append(Rad)
    
    else:
        
        Rad = MMP *2*r
        #Csl_rad.append(Rad)
    
    
    cv2.circle(img,(int(xc),int(yc)),5,(0,0,255),-1) # original laser circle coordinates
    cv2.circle(new_image,(int(xc),int(yc)),int(r),(0,0,0),2) # original laser circle coordinates

    Angle = []  
    pos_Angle = []
    for ib in range(0,len(xy_coords)):
        sb = xy_coords[ib]
        calcu_rad = int(dist.euclidean((xc,yc),(sb[0],sb[1])))
        combine_rad.append(calcu_rad)
        

        #cv2.circle(img,(sb[0],sb[1]),2,(100,255,150),-1) # original laser circle coordinates
        #cv2.line(img, (int(xc),int(yc)) ,(sb[0],sb[1]), (0,0,255), 2) #line connecting center and circle edges
        
        
        angle_radians = math.atan2(((sb[1]-yc)), ((sb[0]-xc)))

# Convert the angle to degrees
        angle_degrees = math.degrees(angle_radians)
        
        Angle.append(int(angle_degrees)) 
        if angle_degrees < 0 :
            angle_deg_pos = 360+ angle_degrees
        else:
            angle_deg_pos = angle_degrees
           
        pos_Angle.append(int(angle_deg_pos)) 

        
        angle_sorting = np.sort(Angle)
        angle_int = angle_sorting.astype(int)
        
    XY_COR = np.array(xy_coords)
    COMBINE_RAD = np.array(combine_rad)
    ANGLE = np.array(Angle)
    pos_ANGLE = np.array(pos_Angle)
    combine_mat = np.column_stack((pos_ANGLE, COMBINE_RAD, XY_COR,ANGLE))
    sort_combine_mat = combine_mat[combine_mat[:,0].argsort()]   
    del sb
    rad_23_67 = []
    rad_67_112 = []
    rad_112_157 = []
    rad_157_202 = []
    rad_202_247 = []
    rad_247_292 = []
    rad_292_337 = []
    rad_337_23 = []
    for ic in range(0,len(xy_coords)):
        an = combine_mat[ic,0]
        sbx = combine_mat[ic,2]
        sby = combine_mat[ic,3]
        # if an < 25 or an > 340: # show all the negative angle
        #     sbx = combine_mat[ic,2]
        #     sby = combine_mat[ic,3]
        #     cv2.circle(img,(sbx,sby),2,(100,255,150),-1)
        
        if 23 < an < 67 : # show all the negative angle
            #sbx = combine_mat[ic,2]
            #sby = combine_mat[ic,3]
            
            rad_23_67.append(combine_mat[ic,1])
            cv2.circle(img,(sbx,sby),2,(0,0,0),-1)
            #cv2.line(img, (int(xc),int(yc)) ,(sbx,sby), (0,0,0), 2)
        
        elif 67 < an < 112 : # show all the negative angle
            #sbx = combine_mat[ic,2]
            #sby = combine_mat[ic,3]
            rad_67_112.append(combine_mat[ic,1])
            cv2.circle(img,(sbx,sby),2,(255,255,255),-1)
            
        elif 112 < an < 157: # show all the negative angle
            #sbx = combine_mat[ic,2]
            #sby = combine_mat[ic,3]
            rad_112_157.append(combine_mat[ic,1])
            cv2.circle(img,(sbx,sby),2,(255,0,0),-1)
            
        elif 157 < an < 202: # show all the negative angle
            #sbx = combine_mat[ic,2]
            #sby = combine_mat[ic,3]
            rad_157_202.append(combine_mat[ic,1])
            cv2.circle(img,(sbx,sby),2,(0,255,0),-1)
            
        elif 202 < an < 247: # show all the negative angle
            #sbx = combine_mat[ic,2]
            #sby = combine_mat[ic,3]
            rad_202_247.append(combine_mat[ic,1])
            cv2.circle(img,(sbx,sby),2,(0,0,255),-1)
            
        elif 247 < an < 292: # show all the negative angle
            #sbx = combine_mat[ic,2]
            #sby = combine_mat[ic,3]
            rad_247_292.append(combine_mat[ic,1])
            cv2.circle(img,(sbx,sby),2,(100,255,250),-1)
            
        elif 292 < an < 337: # show all the negative angle
            #sbx = combine_mat[ic,2]
            #sby = combine_mat[ic,3]
            rad_292_337.append(combine_mat[ic,1])
            cv2.circle(img,(sbx,sby),2,(255,0,150),-1)
            
        elif an < 23 or an > 337: # show all the negative angle
            #sbx = combine_mat[ic,2]
            #sby = combine_mat[ic,3]
            rad_337_23.append(combine_mat[ic,1])
            cv2.circle(img,(sbx,sby),2,(100,255,170),-1)
    
    rad_23 = np.mean(rad_23_67)
    rad_67 = np.mean(rad_67_112)
    rad_112 = np.mean(rad_112_157)
    rad_157 = np.mean(rad_157_202)
    rad_202 = np.mean(rad_202_247)
    rad_247 = np.mean(rad_247_292)
    rad_292 = np.mean(rad_292_337)
    rad_337 = np.mean(rad_337_23)
        
    end_point_23_67 = (
            int(xc + rad_23 * math.cos(math.radians(45))),
            int(yc - rad_23 * math.sin(math.radians(45)))
        ) 
    cv2.line(img, (int(xc),int(yc)), end_point_23_67 , (0,0,255), 1)
    dia_23 = ((2* rad_23)/PPM)
    cv2.putText(img, "Dia 4 = {:.1f}mm".format(dia_23),(end_point_23_67[0],end_point_23_67[1]), cv2.FONT_HERSHEY_SIMPLEX,0.65, (0, 0, 0), 2)
    cv2.line(new_image, (int(xc),int(yc)), end_point_23_67 , (0,0,255), 2)
    cv2.putText(new_image, "Dia 4",(end_point_23_67[0],end_point_23_67[1]), cv2.FONT_HERSHEY_SIMPLEX,0.65, (0, 0, 0), 2)
    cv2.circle(new_image,(end_point_23_67[0],end_point_23_67[1]),5,(0,0,255),-1)


    
    end_point_67_112 = (
            int(xc + rad_67 * math.cos(math.radians(90))),
            int(yc - rad_67 * math.sin(math.radians(90)))
        ) 
    cv2.line(img, (int(xc),int(yc)), end_point_67_112 , (0,0,255), 1)
    dia_67 = ((2* rad_67)/PPM)
    cv2.putText(img, "Dia 3= {:.1f}mm".format(dia_67),(end_point_67_112[0],end_point_67_112[1]), cv2.FONT_HERSHEY_SIMPLEX,0.65, (0, 0, 0), 2)
    
    cv2.line(new_image, (int(xc),int(yc)), end_point_67_112 , (0,0,255), 2)
    cv2.putText(new_image, "Dia 3",(end_point_67_112[0],end_point_67_112[1]-10), cv2.FONT_HERSHEY_SIMPLEX,0.65, (0, 0, 0), 2)
    cv2.circle(new_image,(end_point_67_112[0],end_point_67_112[1]),5,(0,0,255),-1)

    
    end_point_112_157 = (
            int(xc + rad_112 * math.cos(math.radians(135))),
            int(yc - rad_112 * math.sin(math.radians(135)))
        ) 
    cv2.line(img, (int(xc),int(yc)), end_point_112_157 , (0,0,255), 1)
    dia_112 = ((2* rad_112)/PPM)
    cv2.putText(img, "Dia 2 = {:.1f}mm".format(dia_112),(end_point_112_157[0],end_point_112_157[1]), cv2.FONT_HERSHEY_SIMPLEX,0.65, (0, 0, 0), 2)
    
    cv2.line(new_image, (int(xc),int(yc)), end_point_112_157 , (0,0,255), 2)
   
    cv2.putText(new_image, "Dia 2",(end_point_112_157[0],end_point_112_157[1]), cv2.FONT_HERSHEY_SIMPLEX,0.65, (0, 0, 0), 2)
    cv2.circle(new_image,(end_point_112_157[0],end_point_112_157[1]),5,(0,0,255),-1) 
    
    
    end_point_157_202 = (
            int(xc + rad_157 * math.cos(math.radians(180))),
            int(yc - rad_157 * math.sin(math.radians(180)))
        ) 
    cv2.line(img, (int(xc),int(yc)), end_point_157_202 , (0,0,255), 1)
    dia_157 = ((2* rad_157)/PPM)
    cv2.putText(img, "Dia 1 = {:.1f}mm".format(dia_157),(end_point_157_202[0],end_point_157_202[1]), cv2.FONT_HERSHEY_SIMPLEX,0.65, (0, 0, 0), 2)

    cv2.line(new_image, (int(xc),int(yc)), end_point_157_202 , (0,0,255), 2)
    
    cv2.putText(new_image, "Dia 1",(end_point_157_202[0]+10,end_point_157_202[1]), cv2.FONT_HERSHEY_SIMPLEX,0.65, (0, 0, 0), 2)
    cv2.circle(new_image,(end_point_157_202[0],end_point_157_202[1]),5,(0,0,255),-1) 

    
    end_point_202_247 = (
            int(xc + rad_202 * math.cos(math.radians(225))),
            int(yc - rad_202 * math.sin(math.radians(225)))
        ) 
    cv2.line(img, (int(xc),int(yc)), end_point_202_247 , (0,0,255), 1)
    cv2.line(new_image, (int(xc),int(yc)), end_point_202_247 , (0,0,255), 2)
    cv2.circle(new_image,(end_point_202_247[0],end_point_202_247[1]),5,(0,0,255),-1) 

    
    
    end_point_247_292 = (
            int(xc + rad_247 * math.cos(math.radians(270))),
            int(yc - rad_247 * math.sin(math.radians(270)))
        ) 
    cv2.line(img, (int(xc),int(yc)), end_point_247_292 , (0,0,255), 1)
    cv2.line(new_image, (int(xc),int(yc)), end_point_247_292 , (0,0,255), 2)
    cv2.circle(new_image,(end_point_247_292[0],end_point_247_292[1]),5,(0,0,255),-1) 
    

    
    end_point_292_337 = (
            int(xc + rad_292 * math.cos(math.radians(315))),
            int(yc - rad_292 * math.sin(math.radians(315)))
        ) 
    cv2.line(img, (int(xc),int(yc)), end_point_292_337 , (0,0,255), 1)
    cv2.line(new_image, (int(xc),int(yc)), end_point_292_337 , (0,0,255), 2)
    cv2.circle(new_image,(end_point_292_337[0],end_point_292_337[1]),5,(0,0,255),-1) 
    
    end_point_337_23 = (
            int(xc + rad_337 * math.cos(math.radians(0))),
            int(yc - rad_337 * math.sin(math.radians(0)))
        ) 
    cv2.line(img, (int(xc),int(yc)), end_point_337_23 , (0,0,255), 1)
    cv2.line(new_image, (int(xc),int(yc)), end_point_337_23 , (0,0,255), 2)
    cv2.circle(new_image,(end_point_337_23[0],end_point_337_23[1]),5,(0,0,255),-1) 

        
    
    
        # if an < 24:
        #     an_n = 180 + an
        #     new_Angle.append(an_n)
        # ANGLE_n = np.array(new_Angle)
        # combine_mat_n = np.column_stack((ANGLE_n, COMBINE_RAD, XY_COR))
        # sort_combine_mat_n = combine_mat_n[combine_mat_n[:,0].argsort()]     
    '''
        sorting a matrix based on the first column value
        import numpy as np

        A = np.array([0,5,9,8,6])
        B = np.array([1,8,6,9,7])
        C = np.array([7,3,6,5,4])

        D = np.column_stack((A,B,C))


        E = D[D[:, 0].argsort()]
    '''
        
    final_avg_dia =  (dia_23 + dia_67 + dia_112 + dia_157)/4  
    DIA_MET.append(final_avg_dia)
    cv2.putText(img, "Dia = {:.1f}mm".format(final_avg_dia),(100,480), cv2.FONT_HERSHEY_SIMPLEX,0.65, (0, 0, 0), 2)
    cv2.putText(new_image, "Dia 1 = {:.1f}mm".format(dia_157),(100,450), cv2.FONT_HERSHEY_SIMPLEX,0.65, (0, 0, 0), 2)
    cv2.putText(new_image, "Dia 2 = {:.1f}mm".format(dia_112),(100,470), cv2.FONT_HERSHEY_SIMPLEX,0.65, (0, 0, 0), 2)
    cv2.putText(new_image, "Dia 3 = {:.1f}mm".format(dia_67),(100,490), cv2.FONT_HERSHEY_SIMPLEX,0.65, (0, 0, 0), 2)
    cv2.putText(new_image, "Dia 4 = {:.1f}mm".format(dia_23),(100,510), cv2.FONT_HERSHEY_SIMPLEX,0.65, (0, 0, 0), 2)
    
    cv2.putText(new_image, "Avg Dia = {:.1f}mm".format(final_avg_dia),(100,530), cv2.FONT_HERSHEY_SIMPLEX,0.65, (0, 0, 0), 2)


    
    


    k=cv2.imread('a.jpg',img)
    n=cv2.imread('b.jpg',new_red)
    plt.imshow(k)
    plt.imshow(n)
    # cv2.imshow('a.jpg',img)
    # cv2.imshow('b.jpg',new_image)
    # cv2.waitKey()
    # cv2.destroyAllWindows() 


       
    
#     cv2.circle(img,(int(xc),int(yc)),int(r),(255,0,0),2) # fit circle
#     cv2.circle(img,(int(xc),int(yc)),5,(255,0,255),-1)  # fit circle radius
#     #cv2.circle(img,(int(column/2),int(row/2)),5,(0,255,255),-1)  # fit circle radius
    
#     cv2.circle(new_red,(int(xc),int(yc)),int(r),(255,0,0),2) # fit circle
#     cv2.circle(img,(int(xc),int(yc)),5,(0,0,255),-1)  # fit circle radius
#     #cv2.circle(new_red,(int(column/2),int(row/2)),5,(0,255,255),-1)  # fit circle radius
    
    
    # ovality calculation
    
    CR = np.array(combine_rad) 
    all_measure_radius = np.sort(CR)
    
    
    # calculate distance between center and the each edge point; if the distance is more than or less than 10 then remove those points 
    index_end = np.where(all_measure_radius>((r+5))) 
    index_start = np.where(all_measure_radius<(r-5))
    
    new_measure_radius = np.delete(all_measure_radius, index_end)  
    new_measure_radius = np.delete(new_measure_radius, index_start) # deleting all calculated radius value which is out of index 
    '''
    A = np.array([1,2,5,9,8,3,7])
    
    B = np.sort(A)
    
    index_end = np.where(B>8)
    index_start = np.where(B<2)
    
    C = B.copy()
    
    new_B = np.delete(B, index_end)
    new_B = np.delete(new_B, index_start) 
#     '''
     #ovality_cal = ((new_measure_radius[-1] - new_measure_radius[0]) /r)*100
    ovality_cal = ((new_measure_radius[-1] - new_measure_radius[0]) /(new_measure_radius[-1] + new_measure_radius[0]))*100
    Ovality_all.append(ovality_cal)
#     cv2.putText(img, "Dia = {:.1f}mm".format(Rad),(100,480), cv2.FONT_HERSHEY_SIMPLEX,0.65, (0, 0, 0), 2)
    cv2.putText(img, "Ovality = {:.1f}%".format(ovality_cal),(100,500), cv2.FONT_HERSHEY_SIMPLEX,0.65, (0, 0, 0), 2)
    cv2.putText(new_image, "Ovality = {:.1f}%".format(ovality_cal),(100,600), cv2.FONT_HERSHEY_SIMPLEX,0.65, (0, 0, 0), 2)
    
    
    cv2.imwrite(output_path_rgb + img_list[ind],img)
    cv2.imwrite(output_path_format + img_list[ind],new_image)
    
#     #cv2.putText(new_red, "{:.1f}mm".format(Rad),(500,500), cv2.FONT_HERSHEY_SIMPLEX,0.85, (255, 255, 255), 4)
#     #cv2.putText(new_red, "{:.1f}percentage".format(ovality_cal),(100,500), cv2.FONT_HERSHEY_SIMPLEX,0.85, (255, 255, 255), 4)
    
#     #cv2.imwrite(output_path_rgb + img_list[ind],img)
#     # cv2.imwrite(output_path_bw + img_list[ind],new_red)
    
    # cv2.imshow('a1.jpg',img)
    # cv2.imshow('b1.jpg',new_image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    


# # Cslrad = np.array(Csl_rad)    

# # np.save('Calculate_dia.npy',Cslrad)
# # length = len(Cslrad)
# # call_diaa = np.mean(Cslrad) 
# # q_101 = np.repeat(call_diaa,length)  

D_mm = np.array(DIA_MET)

# #np.save('pixel_radius.npy',R_pix)
np.save('Dia_mm.npy',D_mm)
# np.save('ovality_percentage',Ovality_all)   


# D_mm = np.load('Dia_mm.npy')
 
# Mean_dia = np.mean(D_mm)
# Oval1 = np.array(Ovality_all)   
# #R1 = np.array(R_pix)
# D1 = np.array(D_mm)
# D11 = D1[0:400]
# #mean_rad_pix = np.mean(D1)
# #length = len(D1)
# length = len(D11)
# q_100 = list(range(0,length))
# mean_cal_diameter = np.repeat(396,length)
# #mean_cal_diameter = np.repeat(152.4,length)
# #mean_cal_rad = np.repeat(mean_rad_pix,length)  

  
# fig = plt.figure(figsize=(10,5))
# plt.ylim(380,410)
# plt.plot(q_100, D11,'black') 
# #plt.plot(q_100, Cslrad,'black') 
# #plt.plot(q_100, mean_cal_diameter,'red')
# #plt.plot(q_100, Oval1,'blue') 
# #plt.plot(q_100,D_mm,'red')
# plt.plot(q_100,mean_cal_diameter,'blue')
# #plt.plot(q_100,mean_cal_rad,'red')
# plt.xlabel('pipe length') 
# plt.ylabel('Diameter') 
# fig.savefig('dia_pix1.jpg', bbox_inches='tight', dpi=150)
# plt.show()
    



# # #plt.ylim(250,350)
# # plt.plot(q_100, Calrad,'black') 
# # plt.plot(q_100, q_101,'red')
# # plt.xlabel('pipe length') 
# # plt.ylabel('Calculated Diameter')
# # plt.title('Diameter calculation with PTZ camera') 
# # plt.grid(True)
# # fig.savefig('dia1_300mm_.jpg', bbox_inches='tight', dpi=150)
# # plt.show()
    
    
   
   