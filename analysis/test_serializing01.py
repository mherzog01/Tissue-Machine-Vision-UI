# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 23:32:03 2020

@author: MHerzo
"""

import numpy as np
import time
import multiprocessing as mp

#TODO Address issue if have multiple threads all waiting for data.  
#     Don't want to have a lock for a resource that 2 child threads and the parent are waiting for
#     If the parent releases the lock and the 2 children start acquiring it, it could keep the parent
#     from getting the lock, which is what we want

a1 = np.ones((1000,1000))
#processes = []
NUM_IMAGES=10

def get_images():
    for i in range(NUM_IMAGES):
        img = i * a1
        yield {'img_num':i,
               'img_size':img.size,
               'sum':np.sum(img),
               'img':img}

def f(d, cmd, results, lock):
    print('Sub - starting')
    while cmd.value != 'stop':
        try:
            lock.acquire()
            if not d:
                continue
            if not 'sum' in d.keys():
                print(f"Error:  'sum' not found in dict keys {d.keys()}")
                continue
            img_sum = d['sum']
            if not 'img' in d.keys():
                print(f"Error:  'img' not found in dict keys {d.keys()}")
                continue
            img = d['img'] 
            if np.sum(img) != img_sum:
                print(f"Image {d['img_num']} sum {np.sum(img)} != expected {img_sum}")
            results.append(d['img_num'])
        finally:
            lock.release()
    print('subproc exiting')

def main_par():
    with mp.Manager() as manager:
        d = manager.dict()
        results = manager.list()
        cmd = manager.Value('s','init')
        lock = manager.Lock()
        #print(dir(lock))
        #print(f'lock.locked()={lock.locked()}')
        p = mp.Process(target=f,args=(d,cmd,results,lock))
        p.start()
        time.sleep(1)
        for img_dict in get_images():
            lock.acquire()
            for k in img_dict:
                d[k] = img_dict[k]
            lock.release()
            print(f"Main:  d={d['img_num']}")
        print('Stop cmd issued')
        cmd.value = 'stop'
        p.join()
        print(f"Main:  results={results}")

if __name__ == '__main__':
    # main_seq()
    main_par()

