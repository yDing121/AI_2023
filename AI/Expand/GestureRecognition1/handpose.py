#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: universe
"""
import cv2
import numpy as np

# 1.模型加载
protoFile = "hand/pose_deploy.prototxt"
weightsFile = "hand/pose_iter_102000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# 2.图片预处理
frame = cv2.imread('images/00000.jpeg')
frameCopy = np.copy(frame)
frameWidth = frame.shape[1]
frameHeight = frame.shape[0]
aspect_ratio = frameWidth/frameHeight
inHeight = 368
inWidth = int(aspect_ratio*inHeight)
inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

# 3.模型前向推理
net.setInput(inpBlob)
output = net.forward()
print(output.shape)

# 4.寻找关键点
points = []
nPoints = 22
threshold = 0.1

for i in range(nPoints):
    
    probMap = output[0, i, :, :]
    probMap = cv2.resize(probMap, (frameWidth, frameHeight))
    min_val,max_val,min_indx,max_indx = cv2.minMaxLoc(probMap)
    
    # 5.绘制关键点和编号
    if max_val > threshold:
        cv2.circle(frameCopy, (max_indx[0], max_indx[1]), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(frameCopy, "{}".format(i), (max_indx[0], max_indx[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

        points.append((max_indx[0], max_indx[1]))
    else:
        print(max_val,i)
        points.append(None)
        
# 6.绘制关键点条线
POSE_PAIRS = [ [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20] ]

for pair in POSE_PAIRS:
    partA = pair[0]
    partB = pair[1]

    if points[partA] and points[partB]:
        cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
        cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

# 保存图片
cv2.imwrite('Keypoints.jpg', frameCopy)
cv2.imwrite('lines.jpg', frame)