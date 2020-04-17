# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 07:45:02 2020

@author: MHerzo
"""

# From https://www.tensorflow.org/guide/saved_model

import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

CUR_MODEL = './Tissue_Vision_UI__20200413095058-2020-04-14/'


filepath = "C:\\Users\\mherzo\\Documents\\GitHub\\tf-models\\app\\images\WIN_20200411_22_55_36_Pro.jpg"
img = tf.keras.preprocessing.image.load_img(filepath, target_size=[224, 224])
plt.imshow(img)
plt.axis('off')

x = tf.keras.preprocessing.image.img_to_array(img)
#x = tf.keras.applications.mobilenet.preprocess_input(
#    x[tf.newaxis,...])

model = tf.saved_model.load(CUR_MODEL)
infer = model.signatures["serving_default"]
print(infer.structured_outputs)

inference = infer(image_bytes=tf.constant(x), key="score_threshold")
#inference = infer(image_bytes=tf.constant(x), key=tf.constant({"score_threshold": "0.8"}))
# labeling = infer(tf.constant(x))[pretrained_model.output_names[0]]

# decoded = imagenet_labels[np.argsort(labeling)[0,::-1][:5]+1]

# print("Result after saving and loading:\n", decoded)

# --------------------------------------------------------------------
# CLI:
#   saved_model_cli show --dir ./autoML/models/Tissue_Vision_UI__20200413095058-2020-04-14/ --tag_set serve --signature_def serving_default
# --------------------------------------------------------------------
# The given SavedModel SignatureDef contains the following input(s):
#   inputs['image_bytes'] tensor_info:
#       dtype: DT_STRING
#       shape: (-1)
#       name: encoded_image_string_tensor:0
#   inputs['key'] tensor_info:
#       dtype: DT_STRING
#       shape: (-1)
#       name: key:0
# The given SavedModel SignatureDef contains the following output(s):
#   outputs['detection_boxes'] tensor_info:
#       dtype: DT_FLOAT
#       shape: (-1, 40, 4)
#       name: detection_boxes:0
#   outputs['detection_classes'] tensor_info:
#       dtype: DT_FLOAT
#       shape: (-1, 40)
#       name: detection_classes:0
#   outputs['detection_classes_as_text'] tensor_info:
#       dtype: DT_STRING
#       shape: (-1, 40)
#       name: detection_classes_as_text:0
#   outputs['detection_multiclass_scores'] tensor_info:
#       dtype: DT_FLOAT
#       shape: (-1, 40, 2)
#       name: detection_multiclass_scores:0
#   outputs['detection_scores'] tensor_info:
#       dtype: DT_FLOAT
#       shape: (-1, 40)
#       name: detection_scores:0
#   outputs['image_info'] tensor_info:
#       dtype: DT_INT32
#       shape: (-1, 6)
#       name: Tile_1:0
#   outputs['key'] tensor_info:
#       dtype: DT_STRING
#       shape: (-1)
#       name: Identity:0
#   outputs['num_detections'] tensor_info:
#       dtype: DT_FLOAT
#       shape: (-1)
#       name: num_detections:0
# Method name is: tensorflow/serving/predict