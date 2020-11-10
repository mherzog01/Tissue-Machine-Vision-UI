import tensorflow as tf
import io
from PIL import Image

input_type = ['image_tensor', 'float_image_tensor', 'encoded_image_string_tensor', 'tf_example'][2]
if input_type == 'image_tensor':
    img_t = tf.zeros((1,100,100,3), dtype=tf.uint8)
elif input_type == 'encoded_image_string_tensor':    
    image = Image.new('RGB', (20, 20))
    byte_io = io.BytesIO()
    image.save(byte_io, 'PNG')
    img_t = [byte_io.getvalue()]

#model = tf.saved_model.load('/home/mherzog01/projects/tissue-defect-ui/gcp/static/models/tissue_boundary/v10_30_2020_01/model/saved_model')
model = tf.saved_model.load('/tmp/cur_export/saved_model')

results = model(img_t)

for k in results.keys():
    try:
        cur_shape = results[k].shape
    except:
        cur_shape = None
    print(k, cur_shape)