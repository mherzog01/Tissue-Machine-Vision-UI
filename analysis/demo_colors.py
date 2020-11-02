# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 08:18:12 2020

@author: MHerzo
"""


import PIL 
from PIL import Image
import imgviz
import glob
import cv2
import numpy as np

#TODO Clean up -- move to a function and make pythonic - remove hardcodes and reduce back and forth between lists and numpy arrays
label_colormap_orig = imgviz.label_colormap(value=200)
LABEL_COLORMAP = []
# Preserve order
for c in label_colormap_orig:
    # Exclude colors too close to the color of tissue [200,200,200]
    # TODO Make soft
    if all(abs(c - np.array([200,200,200])) < np.array([50,50,50])):
        continue
    c_l = list(c)
    if c_l in LABEL_COLORMAP:
        continue
    LABEL_COLORMAP += [c_l]
LABEL_COLORMAP = np.array(LABEL_COLORMAP)

#cv2.namedWindow("main", cv2.WINDOW_NORMAL)

for fn in glob.glob(r'c:\tmp\work1\*.bmp'):
    img = cv2.imread(fn)
    #https://stackoverflow.com/questions/16815194/how-to-resize-window-in-opencv2-python
    img_r = cv2.resize(img,(600,600))
    for idx,color in enumerate(LABEL_COLORMAP[:10]):
        # https://stackoverflow.com/questions/58215094/change-the-color-of-the-cv2-rectangle
        color_cv2 = tuple(int(c) for c in color)
        cv2.rectangle(img_r, (200, idx * 50), (550, idx * 50 + 25), color_cv2, cv2.FILLED)
        cv2.putText(img_r, f"Color={color}", (10, idx * 50 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
    
    cv2.imshow('main',img_r)

    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()
    if key == ord('q'):
        break
