import ezdxf 

# TODO Use Hatch instead?  Pro:  more efficient.  Con:  harder to manage https://ezdxf.readthedocs.io/en/stable/tutorials/hatch.html
# NestFab does not seem to be able to read sheets which are images using its import

pic_x = 4022
pic_y = 3036
img_path = r'C:\Users\mherzo\AppData\Local\Temp\tissue_vision\input_sheet.png'

doc = ezdxf.new('R2000')  # image requires the DXF R2000 format or later
my_image_def = doc.add_image_def(filename=img_path, size_in_pixel=(pic_x, pic_y))
# The IMAGEDEF entity is like a block definition, it just defines the image

msp = doc.modelspace()
# add first image
msp.add_image(insert=(2, 1, 0), size_in_units=(6.4, 3.6), image_def=my_image_def, rotation=0)

doc.saveas(r"c:\tmp\test.dxf")


