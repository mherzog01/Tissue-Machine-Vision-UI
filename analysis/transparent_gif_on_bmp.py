# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 09:56:51 2020

@author: MHerzo
"""

# https://stackoverflow.com/questions/8376359/how-to-create-a-transparent-gif-or-png-with-pil-python-imaging
from PIL import Image, ImageDraw, ImageFont

img = Image.open(r'C:\Tmp\Work4\20200213-154422-Img - Test.bmp')

img_o = Image.new('RGBA', img.size, (255, 0, 0, 0))

draw = ImageDraw.Draw(img_o)

#https://stackoverflow.com/questions/4902198/pil-how-to-scale-text-size-in-relation-to-the-size-of-the-image
fontsize = 16
font = ImageFont.truetype("arial.ttf", fontsize)
draw.ellipse((2025, 2025, 2075, 2075), fill=(255, 0, 0))
# TODO Use grid - numpy?
for i in range(100,img.size[0],100):
    for j in range(100,img.size[1],100):
        draw.text((i,j),f'{i},{j}', fill='black', font=font)

img_o.save(r'C:\Tmp\Work4\test.gif', 'GIF', transparency=0)
#img_o.show('gif')

#img_combined = img.copy()


#img_combined.show('test')


