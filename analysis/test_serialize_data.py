# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 23:32:03 2020

@author: MHerzo

Objective:  Prove that reading shared memory while it is getting written to 
1.  

Key techniques:
1.  Proxy variables
    - Don't overwrite 
    - Value - use .value
    - As add values of a dict, other processes may pick up before you are done
2.  Logging - run in cmd prompt, not Spyder
3.  A crash in the subprocess doesn't propogate to the calling procedure.  Subprocess can crash but main can continue as if nothing is wrong.
4.  Can't just check if locked -- may change state on the next statement!
5.  try:  finally: is your friend -- lets you unlock and otherwise clean up

Q:  Is the manager using shared memory or pickle?
Q:  Would Ray be better?

Articles and Resources:
- Medium - Good intro quote: https://www.cloudcity.io/blog/2019/02/27/things-i-wish-they-told-me-about-multiprocessing-in-python/
- Good list of major issues:  https://softwareengineering.stackexchange.com/a/126941
- MP logging:  https://docs.python.org/3.7/library/multiprocessing.html#logging, 
                https://softwareengineering.stackexchange.com/a/170865
                https://stackoverflow.com/questions/641420/how-should-i-log-while-using-multiprocessing-in-python

"""

import numpy as np
import multiprocessing as mp
import sys
import time
import traceback

# Python version: 3.7.4 (default, Aug  9 2019, 18:34:13) [MSC v.1915 64 bit (AMD64)]
print(f'Python version: {sys.version}')
#num_processes = mp.cpu_count()
num_processes = 1
print(f'# processes {num_processes}')

# https://stackoverflow.com/questions/6832554/multiprocessing-how-do-i-share-a-dict-among-multiple-processes

# Set up set of shared arrays, one for each procesor
#TODO Is "lock" required?
#TODO What cleanup is required on these shared memory variables?
#images = [mp.Array(name=f'img_{i}', lock=True) for i in range(num_processes)]

a1 = np.ones((1000,1000))
#processes = []
NUM_IMAGES=20

def get_images():
    for i in range(NUM_IMAGES):
        img = i * a1
        yield {'img_num':i,
               'img_size':img.size,
               'sum':np.sum(img),
               'img':img}

def evaluator(proc_num, input_data, results, cmd, images_proc, lock):
    
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

    def inc_res(key, msg=None):
        loc_res[key] += 1
        if msg:
            log_msg(msg.format(loc_res[key]))
        return loc_res[key]
        
    def log_msg(msg):
        print(msg)
        
    def log_err(msg):
        log_msg(msg)
        loc_res['errors'].append(msg)
    
    log_msg('Starting')
    expected_keys = set(['img_num', 'img_size', 'sum', 'img'] )
    loc_res['status'] = 'running'
    cur_img_num = -1
    cur_iter = 0
    while True:
        if cmd.value == 'stop':
            log_msg(f'Stopping')
            break
        cur_iter = inc_res('num_iter')
        if cur_iter % 1000 == 0:
            log_msg(f"Iteration {cur_iter}.  Cmd={cmd.value}")
            print(f'Input data = {input_data}')

        if not input_data:
            inc_res('no_data')
            continue

        try:
            lock.acquire()
            diff = expected_keys.symmetric_difference(set(list(input_data.keys())))
            if len(diff) > 0:
                log_err(f'Input missing keys {diff}.  Input={input_data}.')
                continue
            img_num = input_data['img_num']
            img_size = input_data['img_size']
            img = input_data['img']
            img_sum = input_data['sum']
        finally:
            lock.release()

        try:
            if img_num == cur_img_num:
                inc_res('same_img')
                continue
            if img_num in loc_res['images_proc']:
                log_err(f'Input image number is {img_num} already processed and is not the current image {cur_img_num}.  Iter={cur_iter}')
                continue
            cur_img_num = img_num
            if img_size != img.size:
                log_err(f'Image size {img.size} not equal to expected {img_size}')
                continue
            if img_sum != np.sum(img):
                log_err(f'Image size {np.sum(img)} not equal to expected {img_sum}')
                continue
            images_proc.append(img_num)
            inc_res('num_proc')
            loc_res['images_proc'].append(img_num)
            log_msg(f'Processed image {img_num}')
        except Exception as e:
            print(traceback.print_exc())
            log_err(f'Error {e}')
        finally:
            #print(loc_res)
            for k in loc_res:
                results[k] = loc_res[k]
        #break
    loc_res['status'] = 'stopped'
    for k in loc_res:
        results[k] = loc_res[k]
    log_msg('Exiting')


def main_par():

    with mp.Manager() as manager:
        # Set shared values
        input_data = manager.dict()
        results = manager.dict()
        cmd = manager.Value('s','run')
        images_proc = manager.list()
        lock = manager.Lock()
        
#        p1 = mp.Process(target=evaluator, args=(1,input_data, results, cmd, images_proc))
        p1 = mp.Process(target=evaluator, args=(1,input_data, results, cmd, images_proc, lock))
        p1.start()
        time.sleep(1)
        # TODO Where is manager.Pool documented?
        #with manager.Pool(num_processes, initializer=evaluator) as pool:
        start_sec = time.time()
        for img_dict in get_images():
            # `d` is a DictProxy object that can be converted to dict
            print(f"Loading image #{img_dict['img_num']} to shared memory")
            lock.acquire()
            for k in img_dict:
                input_data[k] = img_dict[k]
            lock.release()
        while True:
            timeout = 0
            reached_timeout = ((time.time() - start_sec) > timeout)
            if len(images_proc) > 90 or reached_timeout:
                if reached_timeout:
                    print('In main:  reached timeout')
                print('In main:  stopping')
                cmd.value = 'stop'
                p1.join()
                break
            print(f'In main:  Images processed = {len(images_proc)}')
            time.sleep(0.5)
        print('In main:  Stopped')
        print(results)
        try:
            print('Closing process')
            p1.close()
        except ValueError:
            print('Terminating')
            p1.terminate()


if __name__ == '__main__':
    # main_seq()
    main_par()