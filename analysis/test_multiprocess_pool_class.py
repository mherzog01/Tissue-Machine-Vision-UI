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
import pickle
import traceback

print(f'Module PID={os.getpid()}')

class TestMultiproc():

    def __init__(self, wk_num):
        print('In class init')
        # global, but for each process the module is loaded, so only one global var per process
        self.wk_num = wk_num
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        print('Class init complete')
        
    def log_msg(self, msg):
        print(f'{self.wk_num}: {msg}')

    def init_interpreter(self, model_path):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.interpreter.allocate_tensors()
        self.log_msg('done worker init')
    
    def do_inference(self, img_num, img):
        self.log_msg(f'Processing image {img_num} size={img.size}')
        if img_num == 0:
            # self.log_msg('interpreter: %r' % (hex(id(self.interpreter)),))
            # self.log_msg('input_details: %r' % (hex(id(self.input_details)),))
            # self.log_msg('output_details: %r' % (hex(id(self.output_details)),))
            self.log_msg(f'interpreter: {self.interpreter}')
            self.log_msg(f'input_details: {self.input_details}')
            self.log_msg(f'output_details: {self.output_details}')
        else:
            return
        tstart = time.time()
    
        #TODO Get needed image size from model
        img_resized = cv2.resize(img,(192,192),interpolation=cv2.INTER_AREA )
        
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        # add N dim
        input_data = np.expand_dims(img_resized, axis=0)
        try:
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        except Exception as e:
            print(f'Exception in set_tensor = {e}')
            print(traceback.print_exc())
            return
        
        self.interpreter.invoke()
    
        logit= self.interpreter.get_tensor(self.output_details[0]['index'])
        pred = np.argmax(logit, axis=1)[0]
        logit = list(logit[0])
        duration = time.time() - tstart 
        log_msg(f'Evaluation of pic {idx}, duration={duration} (sec)')
    
        return logit, pred, duration



    def video_reader(self.img_list):
        for idx,img in enumerate(img_list):
            yield (idx,img)
    
    
    def run_proc(self):
        
        print(f'main PID={os.getpid()}')

        #STREAM_NUM = 0
        #cap = cv2.VideoCapture(STREAM_NUM)  # Change only if you have more than one webcams
        
        #TODO ** While waiting for model to resolve, extrapolate location with https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_meanshift/py_meanshift.html#meanshift
    
        print('Getting video input')
        open_video = False
        input_file = r'c:\tmp\video_input.pickle'
        if os.path.exists(input_file):
            try:
                with open(input_file,'rb') as f:
                    img_list = pickle.load(f)
            except Exception as e:
                print(f'Error reading pickled image file.  Error={e}')
                open_video = True
        else:
            open_video = True
        if open_video:
            #TODO make stream_num soft
            print('Initializing video input')
            cap = cv2.VideoCapture(1)  # Change only if you have more than one webcam
            if not cap.isOpened():
                print('ERROR:  Unable to open video input')
                return
            print('Got video')
            img_list = []
            for i in range(10):
                ret, img = cap.read()
                img_list.append(img)
            cap.release()
            with open(input_file,'wb') as f:
                pickle.dump(img_list, f)
    
        print(f'# images={len(img_list)}, size={[img.size for img in img_list]}')
        optimized_graph_def_file = r'C:\Users\mherzo\Documents\GitHub\tf-models\app\autoML\models\tflite-tissue_defect_ui_20200414053330\model.tflite'
    
        # init model once to find out input dimensions
        # interpreter_main = tflite.Interpreter(model_path=optimized_graph_def_file)
        # input_details = interpreter_main.get_input_details()
        # input_w, intput_h = tuple(input_details[0]['shape'][1:3])
    
        #num_test_imgs=100
        # pregenerate random images with values in [0,1]
        #test_imgs = np.random.rand(num_test_imgs, input_w,intput_h).astype(input_details[0]['dtype'])
    
        scores = []
        predictions = []
        it_times = []
    
        tstart = time.time()
        with Pool(processes=1, initializer=worker.init_interpreter, initargs=(optimized_graph_def_file,)) as pool:         # start 4 worker processes
            while True:
                results = pool.starmap(worker.do_inference, video_reader(img_list))
                for result in results:
                    print(result[0][0])
                    #print(len(results[0][0]))
                #ans = input('Press Enter to continue.  Any key to stop: ')
                #if ans:
                #    break
                #scores, predictions, it_times = list(zip(*results))
                #duration =time.time() - tstart
                break
    
        # print('Parent process time for %d images: %.2fs'%(num_test_imgs, duration))
        # print('Inference time for %d images: %.2fs'%(num_test_imgs, sum(it_times)))
        # print('mean time per image: %.3fs +- %.3f' % (np.mean(it_times), np.std(it_times)) )



if __name__ == '__main__':
    tmp = TestMultiproc()
    tmp.run_proc()
