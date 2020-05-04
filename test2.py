import time

class ImgProcStats():
    def __init__(self):
        self.start_time = time.time()
        self.num_images = 0
        self.num_detections = 0
        self.num_results = 0 # Different results, may be null, when previously an object was detected
        self.num_tracks = 0
        self.num_no_tracks = 0
    
    def inc_val(self,name):
        cur_val = getattr(self, name)
        if cur_val is None:
            cur_val = 0
        setattr(self,name,cur_val + 1)

        
ips = ImgProcStats()

ips.inc_val('num_images')
ips.inc_val('num_tracks')
ips.inc_val('num_images')
ips.inc_val('num_images')
ips.inc_val('num_tracks')
ips.inc_val('num_no_tracks')
ips.num_results += 1

ips.__init__()

print(ips.num_images, ips.num_tracks, ips.num_no_tracks, ips.num_results, ips.num_detections)

