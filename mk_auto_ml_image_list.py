# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 10:09:27 2020

@author: MHerzo
"""

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
from os import path as osp
import json

from PIL import Image

GCS_IMAGE_PREFIX = 'gs://tissue-defect-ui-main-bucket/app/images'
IMAGE_LISTS = {'TRAIN':r'C:\Users\mherzo\Documents\GitHub\tf-models\app\models\cur\train_images.txt',
               'TEST':r'C:\Users\mherzo\Documents\GitHub\tf-models\app\models\cur\test_images.txt',
               'VALIDATE':r'C:\Users\mherzo\Documents\GitHub\tf-models\app\models\cur\val_images.txt'}
# IMAGE_LISTS = {'TRAIN':r'C:\Users\mherzo\Documents\GitHub\tf-models\app\models\cur\train_images.txt',
#                'TEST':r'C:\Users\mherzo\Documents\GitHub\tf-models\app\models\cur\train_images.txt',
#                'VALIDATE':r'C:\Users\mherzo\Documents\GitHub\tf-models\app\models\cur\train_images.txt'}
IMAGE_DIR = r'C:\Users\mherzo\Documents\GitHub\tf-models\app\images'
OUTPUT_FILE = r'C:\Users\mherzo\Documents\GitHub\tf-models\app\tissue_vision_annot_List_auto_ml.csv'



# TO-DO replace this with label map
def class_text_to_int(label):
    if label == 'Pointer Tip':
        return 1
    else:
        None


def parse_labelme_json(annot_file):
    parsed_values = {}
    annot_file_path = annot_file if osp.exists(annot_file) else osp.join(IMAGE_DIR,annot_file)
    with open(annot_file_path,'r') as data_file:
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


# =========================================
# File format
# -----------------------------------------
# [set,]image_path[,label,x1,y1,,,x2,y2,,]
# TRAIN,gs://My_Bucket/sample1.jpg,cat,0.125,0.25,,,0.375,0.5,,
# VALIDATE,gs://My_Bucket/sample1.jpg,cat,0.4,0.3,,,0.55,0.55,,
# TEST,gs://My_Bucket/sample1.jpg,dog,0.5,0.675,,,0.75,0.875,,
# [set,]image_path[,label,x1,y1,x2,y1,x2,y2,x1,y2]
# TRAIN,gs://My_Bucket/sample2.jpg,dog,0.56,0.25,0.97,0.25,0.97,0.50,0.56,0.50
# -----------------------------------------
def auto_ml_csv_row(ml_set, image_file, annot_file):
    image_file_path = image_file if osp.exists(image_file) else osp.join(IMAGE_DIR,image_file)
    image = Image.open(image_file_path)
    width, height = image.width, image.height

    img_basename = osp.basename(image_file)

    
    parsed_values = parse_labelme_json(annot_file)

    x1 = parsed_values['xmin'] / width
    x2 = parsed_values['xmax'] / width
    y1 = parsed_values['ymin'] / height
    y2 = parsed_values['ymax'] / height
    ml_class = parsed_values['class']

    csv_row = ','.join([str(o) 
                        for o in 
                        ['', #ml_set, 
                         GCS_IMAGE_PREFIX + '/' + img_basename,
                         ml_class,
                         x1, y1, '', '',
                         x2, y2, '', '']])

    bb_w = parsed_values['xmax'] - parsed_values['xmin']
    bb_h = parsed_values['ymax'] - parsed_values['ymin']
    print('Image={0}, box width={1:0.1f}, height={2:0.1f}'.format(img_basename, bb_w, bb_h))
    print('Image={0:0.1f}, {1:0.1f}, {2:0.1f}, {3:0.1f}, {4:0.3f}, {5:0.3f}'.format(width,height,bb_w,bb_h,bb_w/width, bb_h/height))
    return csv_row


def main():
    num_rows = 0
    with open(OUTPUT_FILE, 'w') as outfile:
        for k in IMAGE_LISTS:
            with open(IMAGE_LISTS[k], 'r') as img_lst_file:
                for image_file_raw in img_lst_file.readlines():
                    num_rows += 1
                    image_file = osp.normpath(image_file_raw).strip()
                    annot_file = osp.splitext(image_file)[0] + '.json'
                    outfile.write( auto_ml_csv_row(k, image_file, annot_file) + '\n')
    print(f'Successfully created the file {OUTPUT_FILE} with {num_rows} rows')

if __name__ == '__main__':
    main()
    print(r'!gsutil cp "C:\Users\mherzo\Documents\GitHub\tf-models\app\tissue_vision_annot_List_auto_ml.csv" gs://tissue-defect-ui-main-bucket/app/automl/')
    