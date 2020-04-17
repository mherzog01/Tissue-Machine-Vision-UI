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

def do_inference(img_idx, img):
    print('Processing image %d'%img_idx)
    print('interpreter: %r' % (hex(id(interpreter)),))
    print('input_details: %r' % (hex(id(input_details)),))
    print('output_details: %r' % (hex(id(output_details)),))

    tstart = time.time()

    img = np.stack([img]*3, axis=2) # replicates layer three time for RGB
    img = np.array([img]) # create batch dimension
    interpreter.set_tensor(input_details[0]['index'], img )
    interpreter.invoke()

    logit= interpreter.get_tensor(output_details[0]['index'])
    pred = np.argmax(logit, axis=1)[0]
    logit = list(logit[0])
    duration = time.time() - tstart 

    return logit, pred, duration

def main_par():
    optimized_graph_def_file = r'C:\Users\mherzo\Documents\GitHub\tf-models\app\autoML\models\tflite-tissue_defect_ui_20200414053330\model.tflite'

    # init model once to find out input dimensions
    interpreter_main = tflite.Interpreter(model_path=optimized_graph_def_file)
    input_details = interpreter_main.get_input_details()
    input_w, intput_h = tuple(input_details[0]['shape'][1:3])

    num_test_imgs=100
    # pregenerate random images with values in [0,1]
    test_imgs = np.random.rand(num_test_imgs, input_w,intput_h).astype(input_details[0]['dtype'])

    scores = []
    predictions = []
    it_times = []

    tstart = time.time()
    with Pool(processes=4, initializer=init_interpreter, initargs=(optimized_graph_def_file,)) as pool:         # start 4 worker processes

        results = pool.starmap(do_inference, enumerate(test_imgs))
        scores, predictions, it_times = list(zip(*results))
    duration =time.time() - tstart

    print('Parent process time for %d images: %.2fs'%(num_test_imgs, duration))
    print('Inference time for %d images: %.2fs'%(num_test_imgs, sum(it_times)))
    print('mean time per image: %.3fs +- %.3f' % (np.mean(it_times), np.std(it_times)) )



if __name__ == '__main__':
    # main_seq()
    main_par()