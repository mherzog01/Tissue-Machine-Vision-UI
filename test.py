from multiprocessing import Process
import os
import test_worker

def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

if __name__ == '__main__':
    info('main line')
    p = Process(target=test_worker.f, args=('bob',))
    p.start()
    p.join()