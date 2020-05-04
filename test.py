# https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/
import cv2
import sys
import time

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if __name__ == '__main__' :

    # Set up tracker.
    # Instead of MIL, you can also use

    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

    # Read video
    #video = cv2.VideoCapture("videos/chaplin.mp4")
    video = cv2.VideoCapture(1)
    #video = cv2.VideoCapture(0)

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()
    
    tracker_type = tracker_types[3]
    tracker = cv2.TrackerTLD_create()
    bbox = (320, 296,31,27)

    debug_track_init_img=r'c:\tmp\img_test2.jpg'   
    frame = cv2.imread(debug_track_init_img)
    ok = tracker.init(frame, bbox)
    if not ok:
        print('Unable to init')
        sys.exit()

    debug_track_init_img2=r'c:\tmp\test2.jpg'
    # if not osp.exists(debug_track_init_img):
    res_img_debug = frame.copy()
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(res_img_debug, p1, p2, (255,0,0), 2, 1)
    cv2.putText(res_img_debug, f"First init.  Bbox={bbox}", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
    cv2.imwrite(debug_track_init_img2,res_img_debug)

    num_iter = 0    
    last_frame = frame
    while True:
        
        num_iter += 1
        
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            print('Can''t read video')
            break
        
        # Start timer
        timer = cv2.getTickCount()
        
        if num_iter % 200 == 0:
            # tracker.clear()
            # del tracker
            tracker = cv2.TrackerTLD_create()
            #bbox = cv2.selectROI(last_frame, False)
            time.sleep(0.025)
            ok = tracker.init(last_frame, bbox)
            if not ok:
                print('Unable to init')
            else:
                print('Init')
                #sys.exit()        

        # Update tracker
        ok, bbox = tracker.update(frame)
        last_frame = frame.copy()

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        else :
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
    
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            cv2.destroyAllWindows()
            break
    cv2.destroyAllWindows()
