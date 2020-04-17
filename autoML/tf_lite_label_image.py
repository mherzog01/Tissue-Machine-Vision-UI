# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""label_image for tflite."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np

from PIL import Image

import tensorflow as tf # TF2
from object_detection.utils import visualization_utils as vis_util


def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]


def get_output(interpreter, output_details, key):
    output = interpreter.get_tensor(output_details[key]['index'])    
    output = output.squeeze()
    return output

# C:\Users\mherzo\Documents\GitHub\tf-models\models\research\object_detection\object_detection_tutorial.ipynb
def show_inference(output_dict, img):
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = np.array(img)
  # Actual detection.
  #output_dict = run_inference_for_single_image(model, image_np)
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['boxes'],
      output_dict['classes'],
      output_dict['scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=8)

  display(Image.fromarray(image_np))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-i',
      '--image',
      # default='/tmp/grace_hopper.bmp',
      default=r'C:\Users\mherzo\Documents\GitHub\tf-models\app\images\WIN_20200411_22_55_39_Pro.jpg',
      help='image to be classified')
  parser.add_argument(
      '-m',
      '--model_file',
      # default='/tmp/mobilenet_v1_1.0_224_quant.tflite',
      default=r'C:\Users\mherzo\Documents\GitHub\tf-models\app\autoML\models\tflite-tissue_defect_ui_20200414053330\model.tflite',
      help='.tflite model to be executed')
  parser.add_argument(
      '-l',
      '--label_file',
#      default='/tmp/labels.txt',
      default=r'C:\Users\mherzo\Documents\GitHub\tf-models\app\ui_label_map.pbtxt',
      help='name of file containing labels')
  parser.add_argument(
      '--input_mean',
      default=127.5, type=float,
      help='input_mean')
  parser.add_argument(
      '--input_std',
      default=127.5, type=float,
      help='input standard deviation')
  args = parser.parse_args()

  interpreter = tf.lite.Interpreter(model_path=args.model_file)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # check the type of the input tensor
  floating_model = input_details[0]['dtype'] == np.float32

  # NxHxWxC, H:1, W:2
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]
  img = Image.open(args.image).resize((width, height))

  # add N dim
  input_data = np.expand_dims(img, axis=0)

  if floating_model:
    input_data = (np.float32(input_data) - args.input_mean) / args.input_std

  interpreter.set_tensor(input_details[0]['index'], input_data)

  interpreter.invoke()

  output_boxes = get_output(interpreter, output_details, 0)
  class_ids = get_output(interpreter, output_details, 1)
  scores = get_output(interpreter, output_details, 2)
  
  # Output of quantized COCO SSD MobileNet v1 model
  #https://www.tensorflow.org/lite/models/object_detection/overview#starter_model
  #
  # Index	Name	Description
  # 0	Locations	Multidimensional array of [10][4] floating point values between 0 and 1, the inner arrays representing bounding boxes in the form [top, left, bottom, right]
  # 1	Classes	Array of 10 integers (output as floating point values) each indicating the index of a class label from the labels file
  # 2	Scores	Array of 10 floating point values between 0 and 1 representing probability that a class was detected
  # 3	Number and detections	Array of length 1 containing a floating point value expressing the total number of detection results

  #num_detected = output_details[3]
  # For now, just take the first object detected
  THRESHOLD=0.7
  good_entries = scores >= THRESHOLD
  labels = load_labels(args.label_file)
  
  for box in output_boxes(good_entries):
    show_inference()
    if floating_model:
      print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
    else:
      print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))