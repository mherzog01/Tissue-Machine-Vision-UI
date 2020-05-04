# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 23:32:03 2020

@author: MHerzo

"""

import numpy as np
import sys
import time
import tensorflow as tf # TF2
import cv2
#from object_detection.utils import visualization_utils as vis_util
import datetime
import pickle

# Model eval parameters
THRESHOLD = 0.1
#EVAL_FREQ = 0.5 # Time between model evaluations, in seconds

class EvalAutoML():
    
    def __init__(self):
        
        self.model_file = r'C:\Users\mherzo\Documents\GitHub\tf-models\app\autoML\models\tflite-tissue_defect_ui_20200414053330\model.tflite'

        # -------------------
        # Initialize model
        # -------------------
        self.log_msg('Initizliaing model')
        self.interpreter = tf.lite.Interpreter(model_path=self.model_file)
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
        
    def get_images(self):        
        with open(r'c:\tmp\video_input.pickle', 'rb') as f:
            image_list = pickle.load(f)
            while True:
                for i, image_np in enumerate(image_list):
                    yield {'img_num':i,
                           #'img_size':image_np.size,
                           #'sum':np.sum(img),
                           'img':image_np}        
    

    def get_output(self, key):
        output = self.interpreter.get_tensor(self.output_details[key]['index'])    
        output = output.squeeze()
        return output
    
    def evaluator(self, input_data):
        
        expected_keys = set(['img_num', 'img'] )
    
        img_num = input_data['img_num']
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
        
        self.interpreter.set_tensor(self.input_details[0]['index'], model_input_data)

        invoke_start_ns = time.time_ns()        
        self.interpreter.invoke()
        invoke_end_ns = time.time_ns()        
        
        boxes = self.get_output(0)
        class_ids = self.get_output(1)
        scores = self.get_output(2)
    
        eval_end_ns = time.time_ns()
        # Sec in ns = 10**9.  Sec in ms = 10**3
        eval_ms = round((eval_end_ns - eval_start_ns) / 1000000)
        invoke_ms = round((invoke_end_ns - invoke_start_ns) / 1000000)
        
        print(f'Img {img_num}, Eval time={eval_ms} ms.  Invoke={invoke_ms}.')
    
        results = dict()
        results['img_num'] = img_num
        results['boxes'] = boxes
        results['classes'] = class_ids
        results['scores'] = scores
        #self.log_msg('Exiting')
        return results
    
    
    
    
    def main_par(self):
        # Process images
        for img_dict in self.get_images():
            results = self.evaluator(img_dict)


if __name__ == '__main__':

    # Python version: 3.7.4 (default, Aug  9 2019, 18:34:13) [MSC v.1915 64 bit (AMD64)]
    print(f'Python version: {sys.version}')
    
    eam = EvalAutoML()
    eam.main_par()