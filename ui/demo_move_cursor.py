# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 20:55:37 2020

@author: MHerzo
"""


import win32api
from time import sleep

#https://stackoverflow.com/questions/13576732/move-mouse-with-python-on-windows-7
def move_to(x,y):
    win32api.SetCursorPos((x,y))

def test_move(x_addl, y_addl):
    new_x, new_y = orig_x + x_addl, orig_y + y_addl
    print(new_x, new_y)
    move_to(new_x, new_y)    
    sleep(0.0001)
    
orig_x, orig_y = win32api.GetCursorPos()    
print(orig_x,orig_y)
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
        
    
        