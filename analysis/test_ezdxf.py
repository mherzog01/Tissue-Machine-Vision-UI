import ezdxf 
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib_backend import MatplotlibBackend
import matplotlib.pyplot as plt
from PIL import Image

# TODO Use Hatch instead?  Pro:  more efficient.  Con:  harder to manage https://ezdxf.readthedocs.io/en/stable/tutorials/hatch.html

pic_x = 4022
pic_y = 3036

doc = ezdxf.new('R2000')  # image requires the DXF R2000 format or later
#doc = ezdxf.readfile(r'c:\tmp\work4\tissue sheet03.dxf')
msp = doc.modelspace()  # adding entities to the model space


# 0. polyline path - picture boundary
# msp.add_lwpolyline([
#     (0,0),
#     (500, 0),
#     (500, 300),
#     (0, 300),
#     (0,0)])

#1. polyline path - tissue
points = [
    (20, 30),
    (100,50),
    (240, 30),
    (240, 210),
    (20, 210),
    (20, 30)
]
msp.add_lwpolyline(points)

# 2. polyline path - defect
msp.add_lwpolyline([
    (40, 60),
    (40, 90),
    (80, 90),
    (40, 60)
])


# 2. polyline path - defect
msp.add_lwpolyline([
    (140, 60),
    (140, 90),
    (180, 90),
    (140, 60)
])


fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ctx = RenderContext(doc)
out = MatplotlibBackend(ax)
layout = ezdxf.layouts.Layout(msp,doc)
Frontend(ctx, out).draw_layout(layout, finalize=True)
tmpfile = r'c:\tmp\test.png'
fig.savefig(tmpfile, dpi=300)
del fig

img = Image.open(tmpfile)
img.show()

#doc.saveas(r"c:\tmp\test.dxf")
doc.saveas(r'\\SOMVDIPMN02005\c$\Users\MHerzo\AppData\Local\Temp\tissue_vision\test_sheet.dxf')


