# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 23:32:03 2020

@author: MHerzo

"""

import numpy as np
import sys
import time
import traceback
import argparse
import tensorflow as tf # TF2
import cv2
#from object_detection.utils import visualization_utils as vis_util
import datetime
import pickle

# Video stream parameters
MAX_TRIES = 10
#TODO See below for STREAM_NUM local defn
#STREAM_NUM = 1

# Model eval parameters
THRESHOLD = 0.1
#EVAL_FREQ = 0.5 # Time between model evaluations, in seconds

class EvalAutoML():
    
    def __init__(self, args):
        
        self.args = args

        # -------------------
        # Initialize model
        # -------------------
        self.log_msg('Initizliaing model')
        self.interpreter = tf.lite.Interpreter(model_path=self.args.model_file)
        self.interpreter.allocate_tensors()
    
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
    
        # NxHxWxC, H:1, W:2
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]
        self.log_msg01(f'input_details={self.input_details}')
        self.log_msg(f'Init done')
        
    def log_msg(self, msg):
        self.log_msg01(f'{datetime.datetime.now():%Y-%m-%d %H:%M:%S}: {msg}')
        
    def log_msg01(self, msg):
        print(f'{msg}')
        
    def log_err(self, msg):
        self.log_msg(f'Error:  {msg}')
        #loc_res['errors'].append(msg)
    
    def get_images(self):
        
        if True:
            with open(r'c:\tmp\video_input.pickle', 'rb') as f:
                image_list = pickle.load(f)
                while True:
                    for i, image_np in enumerate(image_list):
                        yield {'img_num':i,
                               #'img_size':image_np.size,
                               #'sum':np.sum(img),
                               'img':image_np}        
                return
        #TODO Determine why this isn't getting set by the global variable
        STREAM_NUM = 1
        
        
        # https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/camera.html
        # Set up video stream
        self.log_msg(f'Initializing video stream')
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
        self.log_msg(f'Found video stream {STREAM_NUM} after {num_tries} attempt(s)')

        i = 0
        while True:
            i += 1
            ret, image_np = cap.read()
            image_np = cv2.flip(image_np,-1)
            yield {'img_num':i,
                   #'img_size':image_np.size,
                   #'sum':np.sum(img),
                   'img':image_np}
    

    def get_output(self, key):
        output = self.interpreter.get_tensor(self.output_details[key]['index'])    
        output = output.squeeze()
        return output
    
    def evaluator(self, input_data):
        
        def inc_res(key, msg=None):
            loc_res[key] += 1
            if msg:
                self.log_msg(msg.format(loc_res[key]))
            return loc_res[key]
            
        # Local results summary
        loc_res = { 
            'num_iter' : 0,
            'no_data' : 0,
            'same_img' : 0,
            'images_proc' : [],
            'num_proc' : 0,
            'errors' : [],
            'status' : 'Initializing'
            }
    
        #self.log_msg('Beginning eval')
        expected_keys = set(['img_num', 'img'] )
        loc_res['status'] = 'running'
        cur_img_num = -1
        cur_iter = 0
    
        img_num = input_data['img_num']
        #img_size = input_data['img_size']
        img = input_data['img']
        #img_sum = input_data['sum']
    
        # Output of quantized COCO SSD MobileNet v1 model
        #https://www.tensorflow.org/lite/models/object_detection/overview#starter_model
        #
        # Index	Name	Description
        # 0	Locations	Multidimensional array of [10][4] floating point values between 0 and 1, the inner arrays representing bounding boxes in the form [top, left, bottom, right]
        # 1	Classes	Array of 10 integers (output as floating point values) each indicating the index of a class label from the labels file
        # 2	Scores	Array of 10 floating point values between 0 and 1 representing probability that a class was detected
        # 3	Number and detections	Array of length 1 containing a floating point value expressing the total number of detection results
        eval_start_ns = time.time_ns()
        orig_width, orig_height = img.shape[:2]
        image_np_resized = cv2.resize(img,(round(self.width), round(self.height)),interpolation=cv2.INTER_AREA)
    
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        # add N dim
        model_input_data = np.expand_dims(image_np_resized, axis=0)
        
        #  if floating_model:
        #    model_input_data = (np.float32(input_data) - self.args.input_mean) / self.args.input_std
        
        self.interpreter.set_tensor(self.input_details[0]['index'], model_input_data)
        
        self.interpreter.invoke()
        
        output_boxes = self.get_output(0)
        class_ids = self.get_output(1)
        scores_orig = self.get_output(2)
    
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
        good_boxes = len(boxes) > 0
        
        eval_end_ns = time.time_ns()
        # Sec in ns = 10**9.  Sec in ms = 10**3
        eval_ms = round((eval_end_ns - eval_start_ns) / 1000000)
        
        print(f'Img {img_num}, ', end='')
        if good_boxes:
            print(f'Max score={max(scores):0.1f}, boxes={boxes}', end='')
        else:
            if len(scores_orig) > 0:
                print(f'Max score={max(scores_orig):0.1f}', end='')
            else:
                print(f'No scores found', end='')
        print(f'. Eval time={eval_ms} ms.')
    
        results = dict()
        results['img_num'] = img_num
        results['boxes'] = boxes
        results['classes'] = classes
        results['scores'] = scores
        #self.log_msg('Exiting')
        return results
    
    
    
    
    def main_par(self):
        
        self.log_msg('Entering main_par')
    
        #TODO Don't hardcode labels
        #labels = load_labels(self.args.label_file)
        category_index = {0:{'name':'Pointer Tip'}}
    
        # -------------------------------
        # Process images
        # ------------------------------
        start_time = time.time()
        num_images = 0
        cur_boxes = []
        for img_dict in self.get_images():
            
            results = self.evaluator(img_dict)
            num_images += 1
            cur_boxes = results['boxes']
            classes = results['classes']
            scores = results['scores']
    
            cur_time = time.time()
            time_diff = cur_time - start_time
            if time_diff > 5:
                print(f'Images/sec = {num_images/time_diff:0.0f}')
                start_time = cur_time
                num_images = 0
    
            image_np = img_dict['img']
            good_boxes = len(cur_boxes) > 0
    
            # Visualization of the results of a detection, or reapply old.
            if good_boxes:
                try:
                   # vis_util.visualize_boxes_and_labels_on_image_array(
                   #     image_np,
                   #     cur_boxes,
                   #     classes.astype(int),
                   #     scores,
                   #     category_index,
                   #     min_score_thresh=THRESHOLD,
                   #     #use_normalized_coordinates=True,
                   #     use_normalized_coordinates=False,
                   #     line_thickness=2)
                   pass
                except Exception as e:
                    print(f'Error calling visualizing boxes on image.  Error={e}')
                    print(traceback.print_exc())
    
            # Display output
            cv2.imshow('object detection', cv2.resize(image_np, (400, 300)))
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        cv2.destroyAllWindows()
        self.log_msg('In main:  Stopped')


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
    
    # Python version: 3.7.4 (default, Aug  9 2019, 18:34:13) [MSC v.1915 64 bit (AMD64)]
    print(f'Python version: {sys.version}')
    
    eam = EvalAutoML(args)
    eam.main_par()