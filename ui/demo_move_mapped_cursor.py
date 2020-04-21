# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 20:55:37 2020

@author: MHerzo
"""

import win32api
from time import sleep

import demo_menu

from PyQt5.QtWidgets import QApplication
import sys
import cv2


#https://stackoverflow.com/questions/13576732/move-mouse-with-python-on-windows-7
def move_to(x,y):
    #targ_x, targ_y = map_to_targ(x - orig_x,y - orig_y)
    win32api.SetCursorPos((x,y))
    #win32api.SetCursorPos((targ_x,targ_y))

def test_move(x_addl, y_addl):
    new_x, new_y = orig_x + x_addl, orig_y + y_addl
    print(new_x, new_y)
    move_to(new_x, new_y)    
    sleep(0.0001)

def demo_move():
    for i in range(3):
        # Down
        for j in range(200):
            test_move(0,j)
        # Right
        for j in range(200):
            test_move(j, 200)
        # Up
        for j in range(200,0,-1):
            test_move(200,j)
        # Left
        for j in range(200,0,-1):
            test_move(j,0)

def src_offset(cur_x, cur_y):
    return (cur_x - orig_x, cur_y - orig_y)

# Maps a point relative to the source rectangle to a point in the target rectangle
def map_to_targ(rel_x, rel_y):
    targ_x = (rel_x / src_w * targ_w + targ_rect[0][0]) * scale_factor
    targ_y = (rel_y / src_h * targ_h + targ_rect[0][1]) * scale_factor
    print(f'targ_x,targ_y={targ_x,targ_y}')
    return round(targ_x), round(targ_y)

def cur_pos_util():
    prevPos = (-1,-1)
    while True:
        curPos = win32api.GetCursorPos()
        if prevPos != curPos:
            prevPos = curPos
            print(curPos)
    
            
mouseX,mouseY = (None,None)
def mouse_callback(event,x,y,flags,param):
    global mouseX,mouseY
    if (event == cv2.EVENT_LBUTTONDBLCLK) or (event == cv2.EVENT_LBUTTONDOWN):
        mouseX,mouseY = x,y

#orig_x, orig_y = win32api.GetCursorPos()    
#print(f'Orig x,y = {orig_x,orig_y}')

# Target rectangle
# Upper left corner is does not include the window title, in Windows units, not Qt
# TODO Remove hardcoding
# DPI Virtualization? https://stackoverflow.com/questions/32541475/win32api-is-not-giving-the-correct-coordinates-with-getcursorpos-in-python
targ_rect = [[12,46], [610, 560]]
#scale_factor = 2.0
scale_factor = 1.0

# Set up mapping data
targ_w = targ_rect[1][0] - targ_rect[0][0] + 1
targ_h = targ_rect[1][1] - targ_rect[0][1] + 1


#cur_rect_center = src_offset(*win32api.GetCursorPos())
#targ_x, targ_y = map_to_targ(*cur_rect_center)

use_camera = False
if use_camera:
    STREAM_NUM = 1
    cap = cv2.VideoCapture(STREAM_NUM)  # Change only if you have more than one webcams
    if not cap.isOpened():
        print(f'Video from stream {STREAM_NUM} not available')
        raise ValueError

#demo_menu.main()
app = QApplication(sys.argv)
win = demo_menu.MainWindow()

win.show()
win.raise_()

cv2.namedWindow('Image')
cv2.setMouseCallback('Image',mouse_callback)

try:
    image_np = None
    disp_img_size = True
    while True:
        if use_camera:
            ret, image_np = cap.read()
        else:
            if image_np is None:
                image_np = cv2.imread(r'C:\Users\mherzo\Documents\GitHub\tf-models\app\ui\test_img.jpg')
        if disp_img_size:
            print(f'Image shape = {image_np.shape}')
            src_h, src_w = image_np.shape[0], image_np.shape[1]
            disp_img_size = False
            
        # Display output
        disp_img = image_np
        #disp_img = cv2.resize(image_np, (image_np.shape[1],image_np.shape[0]))
        cv2.imshow('Image', disp_img)
    
        # https://stackoverflow.com/questions/28327020/opencv-detect-mouse-position-clicking-over-a-picture
        k = cv2.waitKey(20) & 0xFF
        if k  == ord('q'):
            break
        elif k == ord('a'):
            if mouseX is None or mouseY is None:
                print(f'Mouse courd {mouseX,mouseY}.  Double click on image and press "a" to perform map.')
            else:
                print(f'Mouse coord:  {mouseX,mouseY}')
                targ_x, targ_y = map_to_targ(mouseX, mouseY)
                move_to(targ_x, targ_y)
finally:        
    cv2.destroyAllWindows()
    win.close()
    #app.exec_()

#demo_move()

#sys.exit(app.exec_())
        
    
        