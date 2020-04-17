# From:
#    https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.py
#    https://stackoverflow.com/questions/51023367/how-to-make-tfrecords-from-json-annotated-images
"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record
  # Create test data:
  python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
from os import path as osp
import io
import tensorflow as tf
import absl
import json

from PIL import Image
from object_detection.utils import dataset_util
#from collections import namedtuple, OrderedDict

flags = absl.app.flags
flags.DEFINE_string('image_list', './models/cur/train_images.txt', 'File containing paths to images to process')
#TODO Support separate outputs to train, val, testdev recordsets
flags.DEFINE_string('output_path', './models/cur/images.record', 'Path to output TFRecord')
#TODO SUpport annotations in a separate dir than iamges
#flags.DEFINE_string('image_dir', './images', 'Path to images')
FLAGS = flags.FLAGS


# TO-DO replace this with label map
def class_text_to_int(label):
    if label == 'Pointer Tip':
        return 1
    else:
        None


# Input:  annotation file
# Output:  dictionary of parsed values
#
# Expected format:
#     {
#   "version": "4.2.7",
#   "flags": {},
#   "shapes": [
#     {
#       "label": "<label>",
#       "points": [
#         [
#           ** All with respect to origin (0,0) at upper-left of image
#           upper_left_x,
#           upper_left_y
#         ],
#         [
#           lower_right_x,
#           lower_right_y
#         ]
#       ],
#       "group_id": null,
#       "shape_type": "rectangle",
#       "flags": {
#         "Not in picture": false,
#         "Not in tissue": false
#       }
#     }
#   ],
#   "imagePath": "imgname.jpg",
#   "imageData": null,
#   "imageHeight": image_height,
#   "imageWidth": image_width
# }
#
def parse_labelme_json(annot_file):
    parsed_values = {}
    with open(annot_file,'r') as data_file:
        data = json.load(data_file)
        width, height = data['imageWidth'], data['imageHeight']
        for item in data["shapes"]:
            shape_type = item['shape_type']
            if shape_type == 'rectangle':
                [upper_left, lower_right] = item['points']
                [xmin, ymin] = upper_left
                [xmax, ymax] = lower_right

                parsed_values['width'] = width
                parsed_values['height'] = height
                parsed_values['class'] = item['label']
                parsed_values['xmin'] = xmin
                parsed_values['ymin'] = ymin
                parsed_values['xmax'] = xmax
                parsed_values['ymax'] = ymax
            else:
                #TODO Support other shapes - e.g. polygons
                print(f'Invalid shape type {shape_type}')
                raise ValueError
    return parsed_values
    #{'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'}


def create_tf_example(image_file, annot_file):
    with tf.io.gfile.GFile(image_file, 'rb') as fid:
        encoded_img = fid.read()
    encoded_img_io = io.BytesIO(encoded_img)
    image = Image.open(encoded_img_io)
    width, height = image.size

    filename = osp.basename(image_file).encode('utf8')
    #TODO Support formats other than jpg
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    
    parsed_values = parse_labelme_json(annot_file)

    #width = parsed_values['width']
    #height = parsed_values['height']
    xmins.append(parsed_values['xmin'] / width)
    xmaxs.append(parsed_values['xmax'] / width)
    ymins.append(parsed_values['ymin'] / height)
    ymaxs.append(parsed_values['ymax'] / height)
    classes_text.append(parsed_values['class'].encode('utf8'))
    classes.append(class_text_to_int(parsed_values['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_img),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
#    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    writer = tf.io.TFRecordWriter(FLAGS.output_path)
    #img_path = FLAGS.image_dir
    output_path = FLAGS.output_path
    with open(FLAGS.image_list) as f:
        for image_file_raw in f.readlines():
            image_file = osp.normpath(image_file_raw).strip()
            annot_file = osp.splitext(image_file)[0] + '.json'
            tf_example = create_tf_example(image_file, annot_file)
            writer.write(tf_example.SerializeToString())

    writer.close()
    print('Successfully created the TFRecords: {}'.format(output_path))

    # ** If using Spyder, need to run this process in a new iPython console each time it's executed
    # https://github.com/abseil/abseil-py/issues/36
    #for name in list(flags.FLAGS):
    #  delattr(flags.FLAGS, name)
    #absl.flags.FLAGS.unparse_flags()

if __name__ == '__main__':
    absl.app.run(main)