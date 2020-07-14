# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 22:10:48 2020

@author: MHerzo
"""


import ezdxf
from ezdxf.addons.dxf2code import entities_to_code, block_to_code

doc = ezdxf.readfile(r'c:\tmp\work4\tissue sheet03.dxf')
msp = doc.modelspace()
source = entities_to_code(msp)

# create source code for a block definition
with open(r'c:\tmp\work4\source.py', mode='wt') as f:
    for block in doc.blocks:
        print(block)
        block_source = block_to_code(block)
        f.write(str(block_source))
        
        # # merge source code objects
        # source.merge(block_source)
    
        # f.write(source.import_str())
        # f.write('\n\n')
        # f.write(source.code_str())
        # f.write('\n')