# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 20:55:37 2020

@author: MHerzo
"""


#import tensorflow as tf
#from PIL import Image

#https://stackoverflow.com/questions/37435369/matplotlib-how-to-draw-a-rectangle-on-image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

im_path = r'C:\Users\mherzo\Documents\GitHub\tf-models\app\images\WIN_20200411_22_55_39_Pro.jpg'
im = Image.open(im_path)
#im = np.array(Image.open('stinkbug.png'), dtype=np.uint8)

box = np.array([0.8435166 , 0.18053222, 0.8895838 , 0.20593423])
width, height = 192, 192
#im = img
x1 = box[0] * width
y1 = box[1] * height
x2 = box[2] * width
y2 = box[3] * height
rect = patches.Rectangle((x1,y1),x2 - x1,y2-y1 ,linewidth=1,edgecolor='r',facecolor='none')

# Create figure and axes
fig,ax = plt.subplots(1)

# Display the image
#ax.imshow(im)

# Create a Rectangle patch
#rect = patches.Rectangle((50,100),40,30,linewidth=1,edgecolor='r',facecolor='none')

# Add the patch to the Axes
ax.add_patch(rect)

plt.show()