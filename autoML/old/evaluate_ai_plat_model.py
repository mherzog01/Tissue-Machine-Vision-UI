# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 13:00:36 2020

@author: MHerzo
"""


from labelme import user_extns
from PIL import Image
import datetime
import numpy as np

def show_status(message, show_time=False):
    if show_time:
        msg = f'{datetime.datetime.now():%Y%m%d %H:%M:%S}: {message}'
    else:
        msg = message
    print(msg)
    

img = Image.open(r'c:\tmp\work1\20200211-143336-Img.bmp')
#img_resized = img.resize((224,224), Image.ANTIALIAS)
#img_resized = img.resize((256,256), Image.ANTIALIAS)
#img_resized_np = (np.asarray(img_resized) / 255).astype(np.float32)
#img_resized_np = np.around(img_resized_np, 3)
#img_resized_np = np.asarray(img_resized)
#instances = [img_resized_np.tolist()]

ipm = user_extns.ImgPredMgr()
if not ipm.cred_set:
    # TODO Get from config file
    cred_path = r'c:\tmp\work1\Tissue Defect UI-ML Svc Acct.json'
    ipm.set_cred(cred_path)
ipm.MODEL = 'test_model'
#ipm.MODEL_VERSION = 'v2020_10_08'
ipm.model_img_size = (224,224)

show_status('Getting features', show_time=True)
#response = ipm.predict_json(instances)

predictions = ipm.predict_imgs([img])

show_status('Processing features', show_time=True)
print(f'{len(predictions)}')
#m_to_p = user_extns.MaskToPolygon(targ_size = img.size)
#num_found = 0
#for mask in ipm.pred_masks_np:
#    pts = m_to_p.get_polygon(mask)
#    print(pts)


