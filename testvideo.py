# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 21:21:00 2020

@author: MHerzo

-----------------------------------

Notes:
    - Possible ways to improve performance
        * Only evaluate if "click"
        * Scan a smaller area
        
ToDo:
    - Label and box blinks on and off, even when object is detected.  
      Thresholds in 'vis_util.visualize_boxes_and_labels_on_image_array' that need to be set?
    - Image is inverted
"""

import argparse
import tensorflow as tf # TF2
import cv2
import numpy as np
from object_detection.utils import visualization_utils as vis_util
#from PIL import Image
import time
import traceback


def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]

def get_output(interpreter, output_details, key):
    output = interpreter.get_tensor(output_details[key]['index'])    
    output = output.squeeze()
    return output


def main(args):

    # Video stream parameters
    MAX_TRIES = 10
    STREAM_NUM = 1
    
    # Model eval parameters
    THRESHOLD = 0.3
    EVAL_FREQ = 0.5 # Time between model evaluations, in seconds
    
    #https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/camera.html
    # Set up video stream
    num_tries = 0
    while True:
        cap = cv2.VideoCapture(STREAM_NUM)  # Change only if you have more than one webcams
        num_tries += 1
        if cap.isOpened():
            break
        else:
            if STREAM_NUM == 1:
                STREAM_NUM=0
                num_tries = 0
                continue
            if num_tries > MAX_TRIES:
                print(f'Unable to open device #{STREAM_NUM}') 
                raise ValueError
    print(f'Found video stream {STREAM_NUM} after {num_tries} attempt(s)')

    labels = load_labels(args.label_file)
    category_index = {0:{'name':'Pointer Tip'}}

    interpreter = tf.lite.Interpreter(model_path=args.model_file)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # check the type of the input tensor
    #floating_model = input_details[0]['dtype'] == np.float32

    # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    # Output of quantized COCO SSD MobileNet v1 model
    #https://www.tensorflow.org/lite/models/object_detection/overview#starter_model
    #
    # Index	Name	Description
    # 0	Locations	Multidimensional array of [10][4] floating point values between 0 and 1, the inner arrays representing bounding boxes in the form [top, left, bottom, right]
    # 1	Classes	Array of 10 integers (output as floating point values) each indicating the index of a class label from the labels file
    # 2	Scores	Array of 10 floating point values between 0 and 1 representing probability that a class was detected
    # 3	Number and detections	Array of length 1 containing a floating point value expressing the total number of detection results

    # Detection
    #last_detect = time.time() - EVAL_FREQ
    last_detect = time.time()
    num_frames = 0
    draw_new_boxes = False
    while True:
        # Read frame from camera
        ret, image_np = cap.read()
        num_frames += 1
        
        cur_time = time.time()
        if cur_time > last_detect + EVAL_FREQ:
            eval_start_ns = time.time_ns()
            orig_width, orig_height = image_np.shape[:2]
            image_np_resized = cv2.resize(image_np,(round(width), round(height)),interpolation=cv2.INTER_AREA)
    
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            # add N dim
            input_data = np.expand_dims(image_np_resized, axis=0)
    
    #  if floating_model:
    #    input_data = (np.float32(input_data) - args.input_mean) / args.input_std
    
            interpreter.set_tensor(input_details[0]['index'], input_data)
            
            interpreter.invoke()
            
            output_boxes = get_output(interpreter, output_details, 0)
            class_ids = get_output(interpreter, output_details, 1)
            scores_orig = get_output(interpreter, output_details, 2)
    
            good_entries = scores_orig >= THRESHOLD
    
            classes = class_ids[good_entries]
            scores = scores_orig[good_entries]

            #TODO Size box to fit on image to display
            boxes = [[round(x1*orig_width),
                      round(y1*orig_height),
                      round(x2*orig_width),
                      round(y2*orig_height)]
                     for [x1,y1,x2,y2] in output_boxes[good_entries]]
            
            boxes = np.array(boxes).astype(int)
            draw_new_boxes = len(boxes) > 0
            
            eval_end_ns = time.time_ns()
            # Sec in ns = 10**9.  Sec in ms = 10**3
            eval_ms = round((eval_end_ns - eval_start_ns) / 1000000)
            frame_rate = num_frames / (cur_time - last_detect) 
            
            if draw_new_boxes:
                print(f'Max score={max(scores)}, boxes={boxes}', end='')
            else:
                if len(scores_orig) > 0:
                    print(f'Max score={max(scores_orig)}', end='')
                else:
                    print(f'No scores found', end='')
            print(f', frame_rate={frame_rate:0.0f}.  Eval time={eval_ms} ms.')
            last_detect = cur_time
            num_frames = 0
            
        # Visualization of the results of a detection, or reapply old.
        if draw_new_boxes:
            try:
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    boxes,
                    classes.astype(int),
                    scores,
                    category_index,
                    #use_normalized_coordinates=True,
                    use_normalized_coordinates=False,
                    line_thickness=2)
            except Exception as e:
                print(f'Error calling visualizing boxes on image.  Error={e}')
                print(traceback.print_exc())
                cv2.destroyAllWindows()
                break

        # Display output
        cv2.imshow('object detection', cv2.resize(image_np, (400, 300)))
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

# Detection
# with detection_graph.as_default():
#     with tf.compat.v1.Session(graph=detection_graph) as sess:
#         while True:
#             # Read frame from camera
#             ret, image_np = cap.read()
#             # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
#             image_np_expanded = np.expand_dims(image_np, axis=0)
#             # Extract image tensor
#             image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
#             # Extract detection boxes
#             boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
#             # Extract detection scores
#             scores = detection_graph.get_tensor_by_name('detection_scores:0')
#             # Extract detection classes
#             classes = detection_graph.get_tensor_by_name('detection_classes:0')
#             # Extract number of detectionsd
#             num_detections = detection_graph.get_tensor_by_name(
#                 'num_detections:0')
#             # Actual detection.
#             (boxes, scores, classes, num_detections) = sess.run(
#                 [boxes, scores, classes, num_detections],
#                 feed_dict={image_tensor: image_np_expanded})
#             # Visualization of the results of a detection.
#             vis_util.visualize_boxes_and_labels_on_image_array(
#                 image_np,
#                 np.squeeze(boxes),
#                 np.squeeze(classes).astype(np.int32),
#                 np.squeeze(scores),
#                 category_index,
#                 use_normalized_coordinates=True,
#                 line_thickness=8)

#             # Display output
#             cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))

#             if cv2.waitKey(25) & 0xFF == ord('q'):
#                 cv2.destroyAllWindows()
#                 break

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

  main(args)