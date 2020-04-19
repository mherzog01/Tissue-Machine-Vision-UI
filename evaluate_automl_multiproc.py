# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 23:32:03 2020

@author: MHerzo

"""

import numpy as np
import multiprocessing as mp
import sys
import time
import traceback
import random
import argparse
import tensorflow as tf # TF2
import cv2
from object_detection.utils import visualization_utils as vis_util
import pickle

# Video stream parameters
MAX_TRIES = 10
#TODO See below for STREAM_NUM local defn
#STREAM_NUM = 1

# Model eval parameters
THRESHOLD = 0.3
EVAL_FREQ = 0.5 # Time between model evaluations, in seconds

#class Worker():
    
def get_output(interpreter, output_details, key):
    output = interpreter.get_tensor(output_details[key]['index'])    
    output = output.squeeze()
    return output

def get_images():
    #TODO Determine why this isn't getting set by the global variable
    STREAM_NUM = 1
    cap = cv2.VideoCapture(STREAM_NUM)  # Change only if you have more than one webcams
    
    # https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/camera.html
    # Set up video stream
    print(f'Initializing video stream')
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

    i = 0
    while True:
        i += 1
        ret, image_np = cap.read()
        yield {'img_num':i,
               #'img_size':image_np.size,
               #'sum':np.sum(img),
               'img':image_np}

#TODO Move into a class
def evaluator(proc_num, 
              input_data, 
              results, 
              cmd, 
              status, 
              input_lock, 
              results_lock,
              args):
    
    def inc_res(key, msg=None):
        loc_res[key] += 1
        if msg:
            log_msg(msg.format(loc_res[key]))
        return loc_res[key]
        
    def log_msg(msg):
        print(f'#{proc_num}: {msg}')
        
    def log_err(msg):
        log_msg(f'Error:  {msg}')
        #loc_res['errors'].append(msg)
    
    # -------------------
    # Initialize model
    # -------------------
    log_msg('Starting')
    interpreter = tf.lite.Interpreter(model_path=args.model_file)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    
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

    log_msg('Model initialized')
    expected_keys = set(['img_num', 'img'] )
    loc_res['status'] = 'running'
    cur_img_num = -1
    cur_iter = 0
    while True:
        if cmd.value == 'stop':
            log_msg(f'Stopping')
            break
        cur_iter = inc_res('num_iter')
        #if cur_iter % 1000 == 0:
        #    log_msg(f"Iteration {cur_iter}.  Cmd={cmd.value}")
        #    print(f'Input data = {input_data}')

        try:
            if not input_data:
                pass
        except:
            log_err('Error:  Input data={input_data}')
        if not input_data:
            inc_res('no_data')
            continue

        try:
            input_lock.acquire()
            diff = expected_keys.symmetric_difference(set(list(input_data.keys())))
            if len(diff) > 0:
                log_err(f'Input missing keys {diff}.  Input={input_data}.')
                continue
            img_num = input_data['img_num']
            #img_size = input_data['img_size']
            img = input_data['img']
            #img_sum = input_data['sum']
        finally:
            input_lock.release()

        good_boxes = False
        try:
            if img_num == cur_img_num:
                inc_res('same_img')
                continue
            if img_num in loc_res['images_proc']:
                log_err(f'Input image number is {img_num} already processed and is not the current image {cur_img_num}.  Iter={cur_iter}')
                continue
            cur_img_num = img_num
            # if img_size != img.size:
            #     log_err(f'Image size {img.size} not equal to expected {img_size}')
            #     continue
            # if img_sum != np.sum(img):
            #     log_err(f'Image size {np.sum(img)} not equal to expected {img_sum}')
            #     continue

            # -----------------------------
            # Process image
            # -----------------------------
            #log_msg(f'Working on img {img_num}')
            status.value = 'running'
            inc_res('num_proc')
            loc_res['images_proc'].append(img_num)

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
            image_np_resized = cv2.resize(img,(round(width), round(height)),interpolation=cv2.INTER_AREA)
    
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            # add N dim
            model_input_data = np.expand_dims(image_np_resized, axis=0)
    
    #  if floating_model:
    #    model_input_data = (np.float32(input_data) - args.input_mean) / args.input_std
    
            interpreter.set_tensor(input_details[0]['index'], model_input_data)
            
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
            good_boxes = len(boxes) > 0
            
            eval_end_ns = time.time_ns()
            # Sec in ns = 10**9.  Sec in ms = 10**3
            eval_ms = round((eval_end_ns - eval_start_ns) / 1000000)
            
            if True:
                print(f'Img {img_num}, ', end='')
                if good_boxes:
                    print(f'Max score={max(scores)}, boxes={boxes}', end='')
                else:
                    if len(scores_orig) > 0:
                        print(f'Max score={max(scores_orig)}', end='')
                    else:
                        print(f'No scores found', end='')
                print(f'. Eval time={eval_ms} ms.')

            #boxes = [f'#{proc_num}:{img_num} x1','{img_num} y1', '{img_num} x2', '{img_num} y2']
            #log_msg(f'Processed image {img_num}')
        except Exception as e:
            print(traceback.print_exc())
            log_err(f'Error {e}')
        finally:
            status.value = 'avail'
            results_lock.acquire()
            if not 'img_num' in results or results['img_num'] < img_num:
               results['img_num'] = img_num
               results['boxes'] = boxes
               results['classes'] = classes
               results['scores'] = scores
            results_lock.release()
        #break
    status.value = 'stopped'
    log_msg('Exiting')




def main_par(cmd_line_args):

    #TODO Don't hardcode labels
    #labels = load_labels(args.label_file)
    category_index = {0:{'name':'Pointer Tip'}}

    with mp.Manager() as manager:
        # One set of results for all processes
        results = manager.dict()
        results_lock = manager.Lock()
        # Set up shared values dictionary - svd
        svd = dict()
        for i in range(num_processes):
            sv = {'input_data': manager.dict(),
                  'cmd': manager.Value('s','run'),
                  'status': manager.Value('s','avail'),
                  'input_lock': manager.Lock()
                  }
            sv['process'] = mp.Process(target=evaluator, args=(i,sv['input_data'], results, sv['cmd'], sv['status'], sv['input_lock'], results_lock, cmd_line_args))
            sv['process'].start()
            svd[i] = sv
            
        # -------------------------------
        # Process images
        # ------------------------------
        start_time = time.time()
        num_images = 0
        cur_boxes = []
        new_boxes = []
        for img_dict in get_images():
            
            # Give image to a worker if one is available
            num_images += 1
            for i in range(num_processes):
                sv = svd[i]
                if sv['status'].value == 'avail':
                    sv['input_lock'].acquire()
                    #print(f'Img dict={img_dict.keys()}')
                    for k in img_dict:
                        sv['input_data'][k] = img_dict[k]
                    sv['input_lock'].release()
                    break
                
            # Check if we got a new box, or should reuse an old box 
            # for the current picture
            results_lock.acquire()
            if 'boxes' in results:
                new_boxes = results['boxes']
            if not np.array_equal(cur_boxes, new_boxes):
                #print(results)
                cur_boxes = new_boxes                
                if 'classes' in results:
                    classes = results['classes']
                if 'scores' in results:
                    scores = results['scores']
            results_lock.release()

            cur_time = time.time()
            time_diff = cur_time - start_time
            if time_diff > 5:
                print(f'Images/sec = {num_images/time_diff:0.0f}')
                start_time = cur_time
                num_images = 0

            image_np = img_dict['img']
            good_boxes = ('classes' in results and 'scores' in results and len(cur_boxes) > 0)

            # Visualization of the results of a detection, or reapply old.
            if good_boxes:
                try:
                    classes = results['classes']
                    scores = results['scores']
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        cur_boxes,
                        classes.astype(int),
                        scores,
                        category_index,
                        #use_normalized_coordinates=True,
                        use_normalized_coordinates=False,
                        line_thickness=2)
                except Exception as e:
                    print(f'Error calling visualizing boxes on image.  Error={e}')
                    print(traceback.print_exc())

            # Display output
            cv2.imshow('object detection', cv2.resize(image_np, (400, 300)))
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        #TODO Add GUI to draw picture and rectangle
        print('In main:  stopping')
        for i in range(num_processes):
            sv = svd[i]
            sv['cmd'].value = 'stop'
            sv['process'].join()
        print('In main:  Stopped')


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
    #num_processes = mp.cpu_count()
    num_processes = mp.cpu_count() - 1
    #num_processes = 1
    print(f'# processes {num_processes}')
    
    # main_seq()
    main_par(args)