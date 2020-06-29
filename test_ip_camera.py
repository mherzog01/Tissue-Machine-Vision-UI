# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 16:40:11 2020

@author: mherzo
"""

import cv2

cap = cv2.VideoCapture("rtsp://admin:@192.168.5.2/media/video_stream")

while True:
    ret, image = cap.read()
    if not ret:
        print("Error reading camera.")
        break
    cv2.imshow("Test", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()