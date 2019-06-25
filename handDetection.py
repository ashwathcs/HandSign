#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 16:29:57 2019

@author: ashwathcs
"""

import numpy as np
import cv2
frame = cv2.imread('hn.jpg')   
cv2.imshow('fr',frame)
blur = cv2.blur(frame,(3,3))
hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
mask2 = cv2.inRange(hsv,np.array([2,50,50]),np.array([15,255,255]))
kernel_square = np.ones((11,11),np.uint8)
kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
  
#morphological processing
dilation = cv2.dilate(mask2,kernel_ellipse,iterations = 1)
erosion = cv2.erode(dilation,kernel_square,iterations = 1)    
dilation2 = cv2.dilate(erosion,kernel_ellipse,iterations = 1)    
filtered = cv2.medianBlur(dilation2,5)
kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
dilation2 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
dilation3 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
median = cv2.medianBlur(dilation2,5)
ret,thresh = cv2.threshold(median,127,255,0)
    
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
max_area=100
ci=0
for i in range(len(contours)):
	cnt=contours[i] 
    area = cv2.contourArea(cnt)
    if(area>max_area):
		   max_area=area
           ci=i
           cnts = contours[ci]
            
hull = cv2.convexHull(cnts)
x,y,w,h = cv2.boundingRect(cnts)
frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
cv2.drawContours(frame,[hull],-1,(255,255,255),2)
 
only_hand = frame[y:y+h,x:x+w]
cv2.imshow('onlyhand',only_hand)
cv2.imshow('hand',frame)
gray = cv2.cvtColor(only_hand,cv2.COLOR_RGB2GRAY)
to_detect = cv2.resize(gray,(28,28))
cv2.imshow('final image',to_detect)
                
