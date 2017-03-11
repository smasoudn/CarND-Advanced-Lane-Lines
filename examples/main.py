# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 20:27:49 2017

@author: Masoud
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from moviepy.editor import VideoFileClip


import myUtils  # NOTE: myUtils.py contains all utilites fundtions (e.g. Line class, color to binary, etc.)



HISTORY_LENGTH = 20  # History for smooth output


# Camera calibration
#######################################################

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('../camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)
    plt.imshow(img)

cv2.destroyAllWindows()

# Draw a sample corner detection
plt.imshow(img)
plt.title('Sample corner detection for camera calibration')
cv2.imwrite("../output_images/01_corner_detection.jpg", img)
plt.show()


# Computing camera calibrtion matrices
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None,None)


# Subplot adjustment
left  = 0.  # the left side of the subplots of the figure
right = 2    # the right side of the subplots of the figure
bottom = 0   # the bottom of the subplots of the figure
top = 1   # the top of the subplots of the figure
wspace = 0.3   # the amount of width reserved for blank space between subplots
hspace = 0.3
plt.subplots_adjust(left, bottom,   right, top, wspace, hspace)


img = cv2.imread("../camera_cal/calibration1.jpg")
dst = cv2.undistort(img, mtx, dist, None, mtx)

plt.figure(1, figsize=(1,2))
plt.subplot(121)
plt.imshow(img)
plt.title("Distorted image")

plt.subplot(122)
plt.imshow(dst)
plt.title("Undistorted image")
plt.show()
cv2.imwrite("../output_images/02_distorted_image.jpg", img)
cv2.imwrite("../output_images/02_undistorted_image.jpg", dst)


  
# Perspective transformation
#######################################################

offset = 300
src_points = np.float32([[212, 720], [1100, 720], [722, 477], [558, 477]])
dst_points = np.float32([[offset, 720], [1280 - offset, 720], [1280-offset, 400], [offset, 400]])

M = cv2.getPerspectiveTransform(src_points, dst_points)
Minv = cv2.getPerspectiveTransform(dst_points, src_points)

# Test the perspective transformation
img = cv2.imread("../test_images/straight_lines2.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
undist_img = cv2.undistort(img, mtx, dist, None, mtx)
warped_img = cv2.warpPerspective(undist_img, M, (undist_img.shape[1], undist_img.shape[0]))


color_binary = myUtils.image2binary(undist_img)
            
# Combine the two binary thresholds
combined_binary = np.zeros_like(color_binary[:,:,0])
combined_binary[(color_binary[:,:,2] == 1) | (color_binary[:,:,1] == 1)] = 1
    
# Perspective transform
warped_binary_img = cv2.warpPerspective(combined_binary, M, combined_binary.shape[::-1])
_, _, _, _, _, color_warp, histogram, out_img = myUtils.findLaneLines(warped_binary_img, flag_display =  0)

# Warp the blank back to original image space using inverse perspective matrix (Minv)
newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 

# Combine the result with the original image
#undist_img = cv2.cvtColor(undist_img, cv2.COLOR_BGR2RGB)
result = cv2.addWeighted(undist_img, 1, newwarp, 0.3, 0)


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.set_title('Stacked thresholds')
ax1.imshow(255*color_binary)
ax2.set_title('Combined S channel and gradient thresholds')
ax2.imshow(combined_binary, cmap='gray')
        
        
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.set_title('Undistorted image')
ax1.imshow(undist_img)
ax2.set_title('Perspective transformed image')
ax2.imshow(warped_img)


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,5))
ax1.set_title('Transformed binary image')
ax1.imshow(warped_binary_img, cmap='gray')
ax2.set_title('Histogram')
ax2.plot(histogram)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.set_title('Line detection')
ax1.imshow(out_img, cmap='gray')
ax2.set_title('Final result')
ax2.imshow(result)

    

cv2.imwrite("../output_images/03_original_img.jpg", img)
cv2.imwrite("../output_images/03_transforme_img.jpg", warped_img)
cv2.imwrite("../output_images/04_color_binary.jpg", 255*color_binary)
cv2.imwrite("../output_images/04_thresholded_img.jpg", 255*combined_binary)  
cv2.imwrite("../output_images/04_transformed_binary_img.jpg", 255*warped_binary_img)
cv2.imwrite("../output_images/04_line_detection.jpg", out_img)
cv2.imwrite("../output_images/04_result.jpg", result)




  
# Main function for frame processing
#######################################################
  
def LaneDetection(img):
    
    # undistort images
    undist_img = cv2.undistort(img, mtx, dist, None, mtx)
    
    color_binary = myUtils.image2binary(undist_img, s_thresh, sx_thresh, ksize)
            
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(color_binary[:,:,0])
    combined_binary[(color_binary[:,:,2] == 1) | (color_binary[:,:,1] == 1)] = 1
    
    # Perspective transform
    warped_img = cv2.warpPerspective(combined_binary, M, combined_binary.shape[::-1])
    
    if myUtils.line_obj.detected == False:        
        left_fit, right_fit, left_curv, right_curv, pos, color_warp, _, _ = myUtils.findLaneLines(warped_img, nwindows, margin, minpix, flag_display = 0)
    else:
        left_fit, right_fit, left_curv, right_curv, pos, color_warp, _ = myUtils.findLaneLinesWithPrevInfo(warped_img, myUtils.line_obj.best_left_fit, myUtils.line_obj.best_right_fit, margin, flag_display = 0)
            
    
    averaged_left_curverad = np.mean(myUtils.line_obj.radius_of_curvature_left, axis=0)
    averaged_right_curverad = np.mean(myUtils.line_obj.radius_of_curvature_right, axis=0)


    
    final_curverature = (averaged_left_curverad  + averaged_right_curverad)/2        

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 

    # Combine the result with the original image
    #undist_img = cv2.cvtColor(undist_img, cv2.COLOR_BGR2RGB)
    result = cv2.addWeighted(undist_img, 1, newwarp, 0.3, 0)

    cv2.putText(result,'Radius of Curvature = ' + str(int(final_curverature)) + ' m', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(result,'Vehicle is ' + str(int(100 * pos)) + ' cm from center', (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
   
    return result
    
    
  
# Main script
#######################################################

images = glob.glob('../test_images/test*.jpg')
 
s_thresh=(170, 255)   # Color min and max threshold
sx_thresh=(30, 255)   # Sobel min and max threshold
ksize = 9             # Choose Sobel filter kernel size
nwindows = 9          # Choose the number of sliding windows
margin = 100          # Set the width of the windows +/- margin
minpix = 50           # Set minimum number of pixels found to recenter window

"""
for im in images:
    img = cv2.imread(im)
    plt.figure()
    plt.imshow(LaneDetection(img))  
"""
video_output = '../output_images/project_video_result2.mp4'
video = VideoFileClip("../project_video.mp4")
fp = video.fl_image(LaneDetection)
fp.write_videofile(video_output, audio=False)
