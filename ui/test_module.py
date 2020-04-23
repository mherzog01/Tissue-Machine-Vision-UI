# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 23:32:03 2020

@author: MHerzo
"""

import win32api
    
scale_factor = None
src_h, src_w = (None,None)
targ_rect = None
scale_factor = None

#https://stackoverflow.com/questions/13576732/move-mouse-with-python-on-windows-7
def move_to(x,y):
    #targ_x, targ_y = map_to_targ(x - orig_x,y - orig_y)
    win32api.SetCursorPos((int(round(x)),int(round(y))))
    #win32api.SetCursorPos((targ_x,targ_y))

def map_to_targ(rel_x, rel_y):
    targ_x = (rel_x / src_w * targ_w + targ_rect[0][1]) * scale_factor
    targ_y = (rel_y / src_h * targ_h + targ_rect[0][0]) * scale_factor
    print(f'targ_x,targ_y={targ_x,targ_y}')
    return round(targ_x), round(targ_y)

def get_center(box):
    # Format of box:  [y1, x1, y2,x2]
    #self.logger.info(f'get_center box={box}')
    [y1, x1, y2, x2] = box
    xc = round(x1 + (x2 - x1) / 2)
    yc = round(y1 + (y2 - y1) / 2)
    #self.logger.info(f'xc,yc={xc},{yc}')
    return (xc,yc)
    
# Entry point from calling procedure
print_once = True
def move_cursor(cur_boxes, p_targ_rect, p_scale_factor, p_src_h, p_src_w):
    # Hack
    global targ_rect, scale_factor, src_h, src_w, targ_w, targ_h
    targ_rect, scale_factor, src_h, src_w = (p_targ_rect, p_scale_factor, p_src_h, p_src_w)
    global print_once
    if print_once:
        print(f'targ_rect={targ_rect}, src_h,src_w = ({src_h},{src_w})')
        print_once = False
    print(f'move_cursor - start:  cur_boxes={cur_boxes}')
    targ_h = targ_rect[1][0] - targ_rect[0][0] + 1
    targ_w = targ_rect[1][1] - targ_rect[0][1] + 1
    
    xc, yc = get_center(cur_boxes[0]) 
    xt, yt = map_to_targ(xc, yc)
    if xt is None or yt is None:
        print(f'Can''t move cursor:  xt,yt None {xt},{yt}')
    else:
        print(f'move_cursor - moving to ({xt},{yt})')
        move_to(xt, yt)
    
num_disp = 0    
def analyze_img(img):
    # global num_disp
    # num_disp += 1
    # if num_disp > 100:
    #     return
    # try:
    #     print(f'Img size={img.size}')
    # except Exception as e:
    #     print(f'Unable to get image size.  Img={img}')
    pass    