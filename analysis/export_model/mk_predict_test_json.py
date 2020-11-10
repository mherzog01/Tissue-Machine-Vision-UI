import numpy as np
import json
import io
from PIL import Image
import base64

input_type = ['image_tensor', 'float_image_tensor', 'encoded_image_string_tensor', 'tf_example'][2]
if input_type == 'image_tensor':
    img_np = np.zeros((1,100,100,3), dtype=np.uint8)
    img_data = img_np.tolist()
    outfile_name = 'test.json'
elif input_type == 'encoded_image_string_tensor':    
    image = Image.new('RGB', (20, 20))
    byte_io = io.BytesIO()
    image.save(byte_io, 'JPEG')
    # Required format:
    #   {"instances": [{"b64": "X5ad6u"}, {"b64": "IA9j4nx"}]}
    #   https://cloud.google.com/ai-platform/prediction/docs/reference/rest/v1/projects/predict#request-body-details
    img_data = {'instances': [{"b64": base64.b64encode(byte_io.getvalue()).decode('ascii')}]}
    outfile_name = 'test_jpg.json'

with open(outfile_name,'w') as f:
     json.dump(img_data, f)