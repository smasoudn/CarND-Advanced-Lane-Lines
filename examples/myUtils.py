# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 12:33:02 2017

@author: Masoud
"""
import numpy as np
import cv2
from collections import deque
import matplotlib.pyplot as plt



class Line():
    def __init__(self, HISTORY_LENGTH=1):
        # was the line detected in the last iteration?
        self.detected = False  
        # polynomial coefficients for the last n fits of the line
        self.recent_left_fitted = deque([], HISTORY_LENGTH)
        self.recent_right_fitted = deque([], HISTORY_LENGTH)                
        #polynomial coefficients averaged over the last n iterations
        self.best_left_fit = None  
        self.best_right_fit = None  
        #radius of curvature of the line for the last n iterations in world units
        self.radius_of_curvature_left = deque([], HISTORY_LENGTH)
        self.radius_of_curvature_right = deque([], HISTORY_LENGTH)
     
        
HISTORY_LENGTH=20
line_obj = Line(HISTORY_LENGTH)


# Color image to binary
#######################################################

def image2binary(img, s_thresh=(170, 255), sx_thresh=(30, 255), ksize = 9):
    img = np.copy(img)
    
    # RGB 2 Gray
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)    
    
    # RGB 2 HLS and separate the S channel
    hls = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)    
    s_channel = hls[:,:,2]

    
    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize = ksize) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    

    return color_binary

        
def findLaneLines(warped_img, nwindows=9, margin=100, minpix=50, flag_display = 1):
       
    # Histogram (for finding the lane lines)    
    histogram = np.sum(warped_img[int(warped_img.shape[0]/2):,:], axis=0)  
    
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((warped_img, warped_img, warped_img))*255     
        
    # Find the peak of the left and right halves of the histogram
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Set height of windows    
    window_height = np.int(warped_img.shape[0]/nwindows)
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped_img.shape[0] - (window+1)*window_height
        win_y_high = warped_img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
    
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    
    # Implement a moving average filter to smooth things and minimize effect of outliers.
    line_obj.recent_left_fitted.append(left_fit)
    line_obj.recent_right_fitted.append(right_fit)
    
    line_obj.best_left_fit = np.mean(line_obj.recent_left_fitted, axis=0)
    line_obj.best_right_fit = np.mean(line_obj.recent_right_fitted, axis=0)
    
        
    # Generate x and y values for plotting
    ploty = np.linspace(0, warped_img.shape[0]-1, warped_img.shape[0] )
    left_fitx = line_obj.best_left_fit[0]*ploty**2 + line_obj.best_left_fit[1]*ploty + line_obj.best_left_fit[2]
    right_fitx =  line_obj.best_right_fit [0]*ploty**2 +  line_obj.best_right_fit [1]*ploty +  line_obj.best_right_fit [2]
                
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
     
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    
    
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    line_obj.radius_of_curvature_left.append(left_curverad)
    line_obj.radius_of_curvature_right.append(right_curverad)
    
    
    #Calculate position of the vehicle in the lane        
    pos = (warped_img.shape[1]/2 - (left_fitx[-1] + right_fitx[-1])/2)*xm_per_pix

       

    # Sanity check
    if abs((right_fitx[0] - left_fitx[0]) - (right_fitx[-1] - left_fitx[-1])) > 400 or abs(left_curverad - right_curverad) > 1000:
        line_obj.detected = False
    else:
        line_obj.detected = True
    
    
    
    # Create an image to draw the lines on the original image
    warp_zero = np.zeros_like(warped_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
 

    if flag_display==1:                    
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        
        
    return left_fit, right_fit, left_curverad, right_curverad, pos, color_warp, histogram, out_img


def findLaneLinesWithPrevInfo(warped_img, left_fit, right_fit, margin=100, flag_display = 1):
    
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((warped_img, warped_img, warped_img))*255     

    nonzero = warped_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Implement a moving average filter to smooth things and minimize effect of outliers.
    line_obj.recent_left_fitted.append(left_fit)
    line_obj.recent_right_fitted.append(right_fit)
    
    line_obj.best_left_fit = np.mean(line_obj.recent_left_fitted, axis=0)
    line_obj.best_right_fit = np.mean(line_obj.recent_right_fitted, axis=0)
    
        
    # Generate x and y values for plotting
    ploty = np.linspace(0, warped_img.shape[0]-1, warped_img.shape[0] )
    left_fitx = line_obj.best_left_fit [0]*ploty**2 + line_obj.best_left_fit [1]*ploty + line_obj.best_left_fit [2]
    right_fitx = line_obj.best_right_fit[0]*ploty**2 + line_obj.best_right_fit[1]*ploty + line_obj.best_right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    
    
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    line_obj.radius_of_curvature_left.append(left_curverad)
    line_obj.radius_of_curvature_right.append(right_curverad)
    
    
    #Calculate position of the vehicle in the lane        
    pos = (warped_img.shape[1]/2 - (left_fitx[-1] + right_fitx[-1])/2)*xm_per_pix

    
    # Sanity check
    if abs((right_fitx[0] - left_fitx[0]) - (right_fitx[-1] - left_fitx[-1])) > 400 or abs(left_curverad - right_curverad) > 1000:
        line_obj.detected = False
    else:
        line_obj.detected = True
        
        
    # Create an image to draw the lines on the original image
    warp_zero = np.zeros_like(warped_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
 

    if flag_display==1:                    
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        
    
    return left_fit, right_fit, left_curverad, right_curverad, pos, color_warp, out_img