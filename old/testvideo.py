# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 21:21:00 2020

@author: MHerzo
"""

import cv2
import numpy as np
from object_detection.utils import visualization_utils as vis_util

#https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/camera.html
# Define the video stream
MAX_TRIES = 10
STREAM_NUM = 1

num_tries = 0
while True:
    cap = cv2.VideoCapture(STREAM_NUM)  # Change only if you have more than one webcams
    num_tries += 1
    if cap.isOpened():
        break
    else:
        if STREAM_NUM == 1:
            STREAM_NUM=0
            num_tries = 0
            continue
        print(f'Unable to open device #{STREAM_NUM}') 
        raise ValueError
print(f'Found video stream {STREAM_NUM} after {num_tries} attempt(s)')

# Detection
while True:
    # Read frame from camera
    ret, image_np = cap.read()
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    boxes = []
    classes = []
    scores = []
    category_index = None
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)

    # Display output
    cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

# Detection
# with detection_graph.as_default():
#     with tf.compat.v1.Session(graph=detection_graph) as sess:
#         while True:
#             # Read frame from camera
#             ret, image_np = cap.read()
#             # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
#             image_np_expanded = np.expand_dims(image_np, axis=0)
#             # Extract image tensor
#             image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
#             # Extract detection boxes
#             boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
#             # Extract detection scores
#             scores = detection_graph.get_tensor_by_name('detection_scores:0')
#             # Extract detection classes
#             classes = detection_graph.get_tensor_by_name('detection_classes:0')
#             # Extract number of detectionsd
#             num_detections = detection_graph.get_tensor_by_name(
#                 'num_detections:0')
#             # Actual detection.
#             (boxes, scores, classes, num_detections) = sess.run(
#                 [boxes, scores, classes, num_detections],
#                 feed_dict={image_tensor: image_np_expanded})
#             # Visualization of the results of a detection.
#             vis_util.visualize_boxes_and_labels_on_image_array(
#                 image_np,
#                 np.squeeze(boxes),
#                 np.squeeze(classes).astype(np.int32),
#                 np.squeeze(scores),
#                 category_index,
#                 use_normalized_coordinates=True,
#                 line_thickness=8)

#             # Display output
#             cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))

#             if cv2.waitKey(25) & 0xFF == ord('q'):
#                 cv2.destroyAllWindows()
#                 break
