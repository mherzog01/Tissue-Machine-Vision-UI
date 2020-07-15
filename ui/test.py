from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy import QtGui
from qtpy import QtWidgets

class C():
    def __init__(self):
        self.calib_factors = {'w':81.6, 'h':81.8}  # Number of pixels in uom
        self.calib_uom = 'cm'  

        
    def pixels_to_uom(self, pixels, dimension):
        if dimension not in self.calib_factors:
            self.logger.error(f'In pixels_to_uom.  Dimension = {dimension} not in {self.calib_factors}')
            return
        return pixels / self.calib_factors[dimension]
    
    def point_pixels_to_uom(self, point_pixels):
        point_uom = []
        dimensions = self.calib_factors.keys()
        if hasattr(point_pixels, 'x') and hasattr(point_pixels,'y'):
            in_vals = [point_pixels.x(), point_pixels.y()]
        else:
            in_vals = point_pixels
        for value, dimension in zip(in_vals, dimensions):
            point_uom += [self.pixels_to_uom(value, dimension)]
        return point_uom
            
if __name__ == '__main__':
    c = C()
    print(c.point_pixels_to_uom((100,200)))
    print(c.point_pixels_to_uom([200,400]))    
    pt = QtCore.QPoint(400,800)
    print(c.point_pixels_to_uom(pt))    
    