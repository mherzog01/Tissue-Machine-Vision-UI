# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 01:35:57 2020

@author: MHerzo
"""

import ezdxf

doc = ezdxf.new('R2000')  # image requires the DXF R2000 format or later
msp = doc.modelspace()  # adding entities to the model space

hatch = msp.add_hatch(color=7, dxfattribs={
    'hatch_style': 0,
    # 0 = nested
    # 1 = outer
    # 2 = ignore
})

# The first path has to set flag: 1 = external
# flag const.BOUNDARY_PATH_POLYLINE is added (OR) automatically
hatch.paths.add_polyline_path([(0, 0), (10, 0), (10, 10), (0, 10)], is_closed=1, flags=1)

# The second path has to set flag: 16 = outermost
hatch.paths.add_polyline_path([(1, 1), (9, 1), (9, 9), (1, 9)], is_closed=1, flags=16)

#hatch.paths.add_polyline_path([(2, 2), (8, 2), (8, 8), (2, 8)], is_closed=1, flags=0)

# The forth path has to set flag: 0 = default, and so on
#hatch.paths.add_polyline_path([(3, 3), (7, 3), (7, 7), (3, 7)], is_closed=1, flags=0)

#doc.saveas(r"c:\tmp\test2.dxf")
doc.saveas(r'\\SOMVDIPMN02005\c$\Users\MHerzo\AppData\Local\Temp\tissue_vision\hatch_demo.dxf')


