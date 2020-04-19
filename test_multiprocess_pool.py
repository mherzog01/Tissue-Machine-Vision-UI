# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 23:32:03 2020

@author: MHerzo
"""

#https://stackoverflow.com/questions/58745187/speedup-tflite-inference-in-python-with-multiprocessing-pool
import numpy as np
import os, time
import tflite_runtime.interpreter as tflite
from multiprocessing import Pool
import cv2

# https://stackoverflow.com/questions/19151/build-a-basic-python-iterator
# class VideoReader():
#     cap = None
#     def __init__(self, low=0, high=0):
#         print('Initializing video input')
#         #TODO make stream_num soft
#         self.cap = cv2.VideoCapture(0)  # Change only if you have more than one webcams
#         print('Initialization done')
#         self.current = low - 1
#         self.high = high

#     def __iter__(self):
#         return self

#     def __next__(self): # Python 2: def next(self)
#         self.current += 1
#         if self.current < self.high or self.high == 0:
#             ret, img = self.cap.read()
#             return self.current, img
#         raise StopIteration
print('Initializing video input')
#TODO make stream_num soft
cap = cv2.VideoCapture(0)  # Change only if you have more than one webcams
print('Initialization done')

def video_reader():
    img = cap.read()
    yield img        

# global, but for each process the module is loaded, so only one global var per process
interpreter = None
input_details = None
output_details = None
def init_interpreter(model_path):
    global interpreter
    global input_details
    global output_details
    interpreter = tflite.Interpreter(model_path=model_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.allocate_tensors()
    print('done init')

def do_inference(idx, img):
    print('Processing image %d'%idx)
    print('interpreter: %r' % (hex(id(interpreter)),))
    print('input_details: %r' % (hex(id(input_details)),))
    print('output_details: %r' % (hex(id(output_details)),))
    tstart = time.time()

    #TODO Get needed image size from model
    img_resized = cv2.resize(img,(192,192),interpolation=cv2.INTER_AREA )
    
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    # add N dim
    input_data = np.expand_dims(img_resized, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    interpreter.invoke()

    logit= interpreter.get_tensor(output_details[0]['index'])
    pred = np.argmax(logit, axis=1)[0]
    logit = list(logit[0])
    duration = time.time() - tstart 
    print(f'Evaluation of pic {idx}, duration={duration} (sec)')

    return logit, pred, duration

def main_par():

    #STREAM_NUM = 0
    #cap = cv2.VideoCapture(STREAM_NUM)  # Change only if you have more than one webcams
    
    #TODO ** While waiting for model to resolve, extrapolate location with https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_meanshift/py_meanshift.html#meanshift

    optimized_graph_def_file = r'C:\Users\mherzo\Documents\GitHub\tf-models\app\autoML\models\tflite-tissue_defect_ui_20200414053330\model.tflite'

    # init model once to find out input dimensions
    interpreter_main = tflite.Interpreter(model_path=optimized_graph_def_file)
    input_details = interpreter_main.get_input_details()
    input_w, intput_h = tuple(input_details[0]['shape'][1:3])

    #num_test_imgs=100
    # pregenerate random images with values in [0,1]
    #test_imgs = np.random.rand(num_test_imgs, input_w,intput_h).astype(input_details[0]['dtype'])

    scores = []
    predictions = []
    it_times = []

    tstart = time.time()
    with Pool(processes=4, initializer=init_interpreter, initargs=(optimized_graph_def_file,)) as pool:         # start 4 worker processes
        results = pool.starmap(do_inference, video_reader(),4)
        print(results)
        print(len(results[0][0]))
        #scores, predictions, it_times = list(zip(*results))
        #duration =time.time() - tstart

    # print('Parent process time for %d images: %.2fs'%(num_test_imgs, duration))
    # print('Inference time for %d images: %.2fs'%(num_test_imgs, sum(it_times)))
    # print('mean time per image: %.3fs +- %.3f' % (np.mean(it_times), np.std(it_times)) )



if __name__ == '__main__':
    # main_seq()
    main_par()