# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 11:32:45 2019

@author: user
"""

import cv2
import numpy as np

def difference(test_img,gen_img,threshold):


    h,w,c = gen_img.shape
    
    residual_img = np.zeros((h,w),'uint8')
    threshold_img = np.zeros((h,w),'uint8')
    
    
    for i in range (0,h):
        for j in range (0,w):
            
            origin_pixel = test_img[i,j,2].astype('int32')
            generate_pixel = gen_img[i,j,2].astype('int32')
            
            residual_red = abs(generate_pixel - origin_pixel)
            
            origin_pixel = test_img[i,j,1].astype('int32')
            generate_pixel = gen_img[i,j,1].astype('int32')
            
            residual_green = abs(generate_pixel - origin_pixel)
            
            origin_pixel = test_img[i,j,0].astype('int32')
            generate_pixel = gen_img[i,j,0].astype('int32')
            
            residual_blue = abs(generate_pixel - origin_pixel)
            
            
            residual_img[i,j] = residual_blue.astype('uint8')

    ret,threshold_img = cv2.threshold(residual_img,threshold,255,cv2.THRESH_BINARY)
    
    #Gray to BGR 
    

    
    
    square_img = cv2.cvtColor(threshold_img, cv2.COLOR_GRAY2BGR)#顯示方塊面積

    contours, hierarchy = cv2.findContours(threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        
        x, y, w, h = cv2.boundingRect(c)
        
        cv2.rectangle(square_img, (x,y), (x+w, y+h), (0, 0, 255),1)
    
    
    
    residual_img = cv2.cvtColor(residual_img, cv2.COLOR_GRAY2BGR)
    threshold_img = cv2.cvtColor(threshold_img, cv2.COLOR_GRAY2BGR)
    
    return residual_img , threshold_img

