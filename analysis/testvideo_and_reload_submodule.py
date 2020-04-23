# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 23:32:03 2020

@author: MHerzo
"""

import cv2
    

def process_image(img):
    # tmp = cv2.resize(image_np, (2,2))
    # tmp_f = cv2.flip(tmp, 1)
    # print(tmp)
    # print(tmp_f)
    # break
    img_new = img
    img_new[:20,:320] = [200,200,200]
    img_new = cv2.flip(img_new, -1)
    # img_new = cv2.flip(img_new,0)
    # img_new = cv2.flip(img_new,1)
    return img_new