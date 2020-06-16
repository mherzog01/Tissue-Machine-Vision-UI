import test_proc
import multiprocessing as mp
import time
import datetime

import os

def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

def f(name):
    info('function f')
    print('hello', name)

if __name__ == '__main__':
    info('main line')
    proc_pointer = mp.Process(target=test_proc.run_test)
    proc_pointer.start()
    #test_proc.run_test()
    
    print(f'In main: looping')
    for i in range(15):
        print(f'In main: {datetime.datetime.now():%H:%M:%S}')
        time.sleep(1)
    print(f'In main: ending')
    proc_pointer.terminate()
    time.sleep(1)
    proc_pointer.close()
    print(f'In main: done')
    