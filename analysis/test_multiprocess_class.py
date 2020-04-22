# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 23:32:03 2020

@author: MHerzo
"""

import numpy as np
import time
import multiprocessing as mp
import logging 
import os

#TODO Address issue if have multiple threads all waiting for data.  
#     Don't want to have a lock for a resource that 2 child threads and the parent are waiting for
#     If the parent releases the lock and the 2 children start acquiring it, it could keep the parent
#     from getting the lock, which is what we want

print(f'Loading module pid={os.getpid()}')
a1 = np.ones((1000,1000))
#processes = []
NUM_IMAGES=10

class TestMultiproc():
    
    def __init__(self):
        self.logger = mp.log_to_stderr(logging.INFO)

    def get_images(self):
        for i in range(NUM_IMAGES):
            img = i * a1
            yield {'img_num':i,
                   'img_size':img.size,
                   'sum':np.sum(img),
                   'img':img}
    
    def f(self, d, cmd, results, lock, tmp):
        self.logger.info('Sub - starting')
        while cmd.value != 'stop':
            try:
                lock.acquire()
                if not d:
                    continue
                if not 'sum' in d.keys():
                    self.logger.info(f"Error:  'sum' not found in dict keys {d.keys()}")
                    continue
                img_sum = d['sum']
                if not 'img' in d.keys():
                    self.logger.info(f"Error:  'img' not found in dict keys {d.keys()}")
                    continue
                img = d['img'] 
                if np.sum(img) != img_sum:
                    self.logger.info(f"Image {d['img_num']} sum {np.sum(img)} != expected {img_sum}")
                results.append(d['img_num'])
            finally:
                lock.release()
        self.logger.info('subproc exiting')
    
    def main_par(self):
        with mp.Manager() as manager:
            d = manager.dict()
            results = manager.list()
            cmd = manager.Value('s','init')
            lock = manager.Lock()
            #self.logger.info(dir(lock))
            #self.logger.info(f'lock.locked()={lock.locked()}')
            p = mp.Process(target=self.f,args=(d,cmd,results,lock, self))
            p.start()
            time.sleep(1)
            for img_dict in self.get_images():
                lock.acquire()
                for k in img_dict:
                    d[k] = img_dict[k]
                lock.release()
                self.logger.info(f"Main:  d={d['img_num']}")
            self.logger.info('Stop cmd issued')
            cmd.value = 'stop'
            p.join()
            self.logger.info(f"Main:  results={results}")

if __name__ == '__main__':
    tmp = TestMultiproc()
    # main_seq()
    tmp.main_par()

