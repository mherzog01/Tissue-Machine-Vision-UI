import time
import datetime

class test_class():
    def __init__(self,
                 run_proc=False):
        if run_proc:
            self.run_test()
        
    def run_test(self):
        num_iter = 100
        out_file_path = r'c:\tmp\test.txt'
        with open(out_file_path,'w', buffering=1) as f:
            print('***Process beginning***',file=f)        
            for i in range(num_iter):
                print(f'{datetime.datetime.now():%H:%M:%S}',file=f)
                time.sleep(1) # Delay for 5 seconds.
            print('***Process exiting***',file=f)        
        
if __name__ == '__main__':
    tc = test_class(True)
    #tc.run_test()