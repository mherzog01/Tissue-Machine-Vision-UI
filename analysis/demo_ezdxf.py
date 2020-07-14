# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 01:35:57 2020

@author: MHerzo
"""

import ezdxf

doc = ezdxf.new('R2000')
msp = doc.modelspace()

points = [(0, 0), (3, 0), (6, 3), (6, 6)]
msp.add_lwpolyline(points)

#doc.saveas(r"c:\tmp\test2.dxf")
doc.saveas(r'\\SOMVDIPMN02005\c$\Users\MHerzo\AppData\Local\Temp\tissue_vision\lwpolyline_demo.dxf')


