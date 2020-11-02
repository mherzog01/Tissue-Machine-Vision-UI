from PIL import Image 

from PIL import ImageDraw

#image = Image.new('RGBA', (400, 400),(0,0,0))
image = Image.new('RGBA', (400, 400))
image_map = image.load()
for i in range(200,300):
    for j in range(100,200):
        image_map[i,:] = (100,100,100,255)
draw = ImageDraw.Draw(image)
for i in range(10):
    color = (i,25 * i,255 - 25 * i)
    trans = i * 20
    draw.ellipse((20 + trans, 20+ trans, 70+ trans, 70+ trans), fill = color, outline =color)
#draw.point((100, 100), 'red')
image.save('test.png') 
image.show()