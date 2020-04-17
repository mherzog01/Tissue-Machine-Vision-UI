# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 21:13:23 2020

@author: MHerzo
"""

#From https://www.linuxjournal.com/content/multiprocessing-python

import multiprocessing
import time
import random
import os
from multiprocessing import Queue

q_img_request = Queue()
q_img_data = Queue()
q_img_eval = Queue()

def img_eval():
    # Want the most current image
    # 1.  Get an entry in the q_img_data queue.  If contains an entry, process it.
    # 2.  Read q_Load q
    while True:
        pass
    time.sleep(random.randint(400,600)/1000)
    q.put(os.getpid())
    print("{0}] Hello!".format(n))

if __name__ == '__main__':
    img_eval_proc = multiprocessing.Process(target=img_eval)
    img_eval_proc.start()
    img_eval_proc.join()

print("All processing ends")
