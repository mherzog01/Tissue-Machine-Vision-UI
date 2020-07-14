# https://stackoverflow.com/questions/40883481/python-logging-in-multiprocessing
import multiprocessing
import logging
import multiprocessing_logging
import time

logging.basicConfig(level=logging.INFO, filename=r'c:\tmp\test.log', filemode='w')
logger = logging.getLogger()
multiprocessing_logging.install_mp_handler(logger)

def worker():
    for i in range(100):
        logger.info("This is logging for TEST1")
        time.sleep(0.1)

def worker2():
    for i in range(100):
        logger.info("This is logging for TEST2")
        time.sleep(0.1)

if __name__ == '__main__':
    p1 = multiprocessing.Process(target=worker)
    p1.daemon = True
    p1.start()

    p2 = multiprocessing.Process(target=worker2)
    p2.daemon = True
    p2.start()

    p1.join()
    p2.join()