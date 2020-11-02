# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 10:44:36 2020

@author: MHerzo
"""

import os
import os.path as osp

from qtpy import QtCore, QtWidgets, QtGui
from qtpy.QtWidgets import QWidget
from qtpy.QtWidgets import QInputDialog
from PyQt5.QtCore import QCoreApplication

from PyQt5.QtWidgets import QApplication

import sys
import glob
import datetime
import numpy as np
import pandas as pd

import tempfile
import shutil

import PIL 
from PIL import Image

import labelme
from labelme import user_extns
from labelme.shape import Shape



#DEV_IMG_DIR = r'C:\Users\mherzo\Documents\Cognex Designer\Projects\TissueInsp2_3\Images'
DEV_IMG_DIR = r'C:\tmp\work1'
DEV_LABEL_DIR = DEV_IMG_DIR

PROD_IMG_DIR = r'D:\Tissue Defect Inspection\Images4'
PROD_LABEL_DIR = PROD_IMG_DIR


def stop_process():
    raise ValueError()
    
    
# Given a dictionary of flags (from a shape), 
# return whether the given key is True or False
def get_flag_value(flag_dict, flag_key):
    return (flag_key in flag_dict and flag_dict[flag_key])
    

run_mode = 'D'
# run_mode = input('[D]ev or [P]rod mode (leave blank to exit)? ')
# if run_mode == "":
#     stop_process()
if run_mode.upper() in 'D':
    IMG_DIR = DEV_IMG_DIR
    LABEL_DIR = DEV_LABEL_DIR
elif run_mode.upper() == 'P':
    IMG_DIR = PROD_IMG_DIR
    LABEL_DIR = PROD_LABEL_DIR
else:
    print('Invalid entry')
    raise ValueError
    
img_files = [f for f in glob.glob(IMG_DIR + r'\*.bmp')]
label_dir = LABEL_DIR

# Assume:
#   - List of img_files contain all images captured for the applicable day
#   - Label_dir has annotation data (.json files) for all files in img_files
#   - Files in img_files correspond to entries in the DB in DB_FILE
#    
# Ensure that Files in img_files correspond to entries in the DB in DB_FILE
df_annot = pd.DataFrame(columns=['image_basename',
                                 'annot_num',
                                 'label',
                                 'group_id',
                                 'not_in_picture',
                                 'not_in_tissue',
                                 'review_recommended',
                                 'rework'])
df_annot.astype({'image_basename':str,
                                 'annot_num':int,
                                 'label':str,
                                 'group_id':int,
                                 'not_in_picture':bool,
                                 'not_in_tissue':bool,
                                 'review_recommended':bool,
                                 'rework':bool})
    
for img_file in img_files:
    
    #img_date = datetime.datetime.strptime(row[0].split(' ')[0],'%Y-%m-%d')
        
    print(f'File {img_file}.')
    label_file = user_extns.imgFileToLabelFileName(img_file, label_dir)
    if not osp.exists(label_file):
        print(f'ERROR:  No label file {label_file} found for image file {img_file}.')
        continue
    img_basename = osp.basename(img_file)
    labelFile = labelme.LabelFile(label_file, loadImage=False)    
    #img_unique_labels = set([shape['label'] for shape in labelFile.shapes])
    #print(f'File {img_file}.  img_unique_labels={img_unique_labels}')
    annot_num = 0
    for shape in labelFile.shapes:
        annot_num += 1 
        flag_dict = shape['flags']
        label = shape['label']
        group_id = shape['group_id']
        not_in_picture = get_flag_value(flag_dict, 'Not in picture')
        not_in_tissue = get_flag_value(flag_dict, 'Not in tissue')
        review_recommended = get_flag_value(flag_dict, 'Review recommended')
        rework = get_flag_value(flag_dict, 'Rework')
        #break
        df_annot.loc[len(df_annot)] = {'image_basename':img_basename,
                                 'annot_num':annot_num,
                                 'label':label,
                                 'group_id':group_id,
                                 'not_in_picture':not_in_picture,
                                 'not_in_tissue':not_in_tissue,
                                 'review_recommended':review_recommended,
                                 'rework':rework}
    if annot_num == 0:
        df_annot.loc[len(df_annot),'image_basename'] = img_basename
               
#https://stackoverflow.com/questions/34275782/how-to-get-desktop-location        
annot_xlsx = osp.join(os.environ["HOMEPATH"],'desktop','Annotations.xlsx')
df_annot['Selected'] = None
df_annot.to_excel(annot_xlsx)    
        
    
#    app = QCoreApplication.instance()
#    if app is None:
#        app = QApplication(sys.argv)
    
    
    