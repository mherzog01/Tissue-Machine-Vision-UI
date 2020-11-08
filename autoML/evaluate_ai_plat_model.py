# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 13:00:36 2020

@author: MHerzo
"""


from labelme import user_extns
from PIL import Image
import datetime
import numpy as np
import base64


def show_status(message, show_time=False):
    if show_time:
        msg = f'{datetime.datetime.now():%Y%m%d %H:%M:%S}: {message}'
    else:
        msg = message
    print(msg)
    

ipm = user_extns.ImgPredMgr()
if not ipm.cred_set:
    # TODO Get from config file
    cred_path = r'c:\tmp\work1\Tissue Defect UI-ML Svc Acct.json'
    ipm.set_cred(cred_path)
    
#ipm.CLOUD_PROJECT = 'tmp-project-1603675946093'
#ipm.MODEL = 'error_demo'

ipm.MODEL = 'test_model'

#ipm.MODEL_VERSION = 'v2020_10_08'

ipm.model_img_size = (224,224)

img = Image.open(r'c:\tmp\work1\20200211-143336-Img.bmp')
img_resized = img.resize(ipm.model_img_size, Image.ANTIALIAS)
#img_resized = img.resize((256,256), Image.ANTIALIAS)

#normalize = True
normalize = False
if normalize:
    img_resized_np = (np.asarray(img_resized) / 255).astype(np.float)
    img_resized_np = np.around(img_resized_np, 4)
else:
    img_resized_np = np.asarray(img_resized).astype(np.uint8)
#img_resized_np = np.zeros((1,ipm.model_img_size[0],ipm.model_img_size[1],3), dtype=int)

instances = img_resized_np.tolist()

show_status('Getting features', show_time=True)
response = ipm.predict_json([instances])

# https://cloud.google.com/ai-platform/training/docs/algorithms/object-detection-start
#with open(r'c:\tmp\work1\20200211-143336-Img.bmp', 'rb') as image_file:
#  encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
#encoded_string = base64.b64decode(img_resized.tobytes())

#image_bytes = {'b64': str(encoded_string)}
#instances = {'image_bytes': image_bytes, 'key': '1'}
#response = ipm.predict_json([instances])

#predictions = ipm.predict_imgs([img])

show_status('Processing features', show_time=True)
print(f'{len(response)}')
#m_to_p = user_extns.MaskToPolygon(targ_size = img.size)
#num_found = 0
#for mask in ipm.pred_masks_np:
#    pts = m_to_p.get_polygon(mask)
#    print(pts)


