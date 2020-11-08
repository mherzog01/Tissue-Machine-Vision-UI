from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection import inputs
import os
import re
import matplotlib.pyplot as plt
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
import tensorflow as tf

def plot_detections(image_np,
                    boxes,
                    classes,
                    scores,
                    category_index,
                    figsize=(12, 16),
                    image_name=None,
                    plt_show=False):
  """Wrapper function to visualize detections.

  Args:
    image_np: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    figsize: size for the figure.
    image_name: a name for the image file.
  """
  image_np_with_annotations = image_np.copy()
  viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_annotations,
      boxes,
      classes,
      scores,
      category_index,
      use_normalized_coordinates=True,
      min_score_thresh=0.8)
  if image_name:
    plt.imsave(image_name, image_np_with_annotations)
  else:
    plt.imshow(image_np_with_annotations)
    if plt_show:
      plt.show()


# By convention, our non-background classes start counting at 1.  Given
# that we will be predicting just one class, we will therefore assign it a
# `class id` of 1.

# TODO Get class names from classes.json
#desired_feature_id = 12
# Model may not like classes that are not continuous integers starting at 1
desired_feature_id = 1
num_classes = 1
#targ_config_dir = r'c:\tmp\work4'
targ_config_dir = 'c:/tmp/work4'
desired_feature = 'Tissue boundary'
checkpoint_path = 'x'

# tfrecord files train and test.
train_record_fname = 'c:/users/mherzo/downloads/tfrecord-00000-of-00001'
test_record_fname = train_record_fname

labelmap_path = targ_config_dir + '/labelmap.pbtxt'

category_index = {desired_feature_id: {'id': desired_feature_id, 'name': desired_feature}}

# Label map in needed format
with open(labelmap_path,'w') as f:
  f.write(f"item {{id: {desired_feature_id} name: '{desired_feature}'}}")

model_img_size = (640, 640)
batch_size = 4
#batch_size = len(train_images_np)
num_batches = 1000

orig_pipeline_config = 'c:/tmp/work4/object_detection/configs/tf2/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.config'

#from object_detection.protos import preprocessor
#from object_detection.core import preprocessor
# https://stackoverflow.com/questions/53383151/tensorflow-object-detection-api-specifying-multiple-data-augmentation-options

# random_horizontal_flip is already in the file
addl_aug = ''

if True:
  addl_aug += 'data_augmentation_options {'
  addl_aug += '  random_patch_gaussian {min_patch_size: 1\n'
  addl_aug += '                         max_patch_size:' + str(model_img_size[0]) + '\n'
  addl_aug += '                         random_coef: 0.5}\n'
  addl_aug += '  }\n'

  addl_aug += 'data_augmentation_options {'
  addl_aug += '  random_distort_color { }\n'
  addl_aug += '  }\n'

  addl_aug += 'data_augmentation_options {'
  addl_aug += '  random_rotation90 { }\n'
  addl_aug += '  }\n'

  addl_aug += 'data_augmentation_options {'
  addl_aug += '  random_vertical_flip { }\n'
  addl_aug += '  }\n'

pipeline_config = os.path.join(targ_config_dir, 'pipeline.config')

# From https://blog.roboflow.com/train-a-tensorflow2-object-detection-model
# write custom configuration file by slotting our dataset, model checkpoint, and training parameters into the base pipeline file

print('Writing custom configuration file')

with open(orig_pipeline_config) as f:
    s = f.read()

with open(pipeline_config, 'w') as f:
    # Image size
    s = re.sub('height: 640',
               f'height: {model_img_size[0]}', s)
    s = re.sub('width: 640',
               f'width: {model_img_size[1]}', s)

    # fine_tune_checkpoint
    s = re.sub('fine_tune_checkpoint: ".*?"',
               'fine_tune_checkpoint: "{}"'.format(checkpoint_path), s)
    # tfrecord files train and test.
    s = re.sub(
        '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/train)(.*?")', 'input_path: "{}"'.format(train_record_fname), s)
    s = re.sub(
        '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/val)(.*?")', 'input_path: "{}"'.format(test_record_fname), s)

    # label_map_path
    s = re.sub(
        'label_map_path: ".*?"', 'label_map_path: "{}"'.format(labelmap_path), s)

    # Set training batch_size.
    s = re.sub('batch_size: [0-9]+',
               'batch_size: {}'.format(batch_size), s)

    # Set training steps, num_steps
    s = re.sub('num_steps: [0-9]+',
               'num_steps: {}'.format(num_batches), s)

    # Set number of classes num_classes.
    s = re.sub('num_classes: [0-9]+',
               'num_classes: {}'.format(num_classes), s)

    # Fine-tune checkpoint type
    s = re.sub(
        'fine_tune_checkpoint_type: "classification"', 'fine_tune_checkpoint_type: "{}"'.format('detection'), s)

    # Resize masks - not needed - is done anyway
    # s = re.sub('resize_masks: false', 'resize_masks: {}'.format('true'), s)

    # Updates per https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/instance_segmentation.md#updating-a-faster-r-cnn-config-file
    # ** Not needed - create_coco_tf_record converts mask boundary to PNG
    # s = re.sub('mask_type: PNG_MASKS', 'mask_type: {}'.format('NUMERICAL_MASKS'), s)

    # Augmentations (from prev cell)
    s = re.sub('optimizer {',
               addl_aug + 'optimizer {', s)

    f.write(s)

configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']

detection_model = model_builder.build(model_config,is_training=True,add_summaries=True)
#print(len(detection_model.trainable_variables))

train_ds = inputs.train_input(configs['train_config'], configs['train_input_config'],
                model_config)

dummy_scores = np.array([1.0], dtype=np.float32)  # give boxes a score of 100%

for features, labels in train_ds.take(1):
  for idx, image in enumerate(features['image']):
    image_np = image.numpy().astype(np.float32) / 255 / 2 + 0.5
    #print(np.histogram(image_np))
    #print(f'{np.min(image_np)} {np.max(image_np)}')
    boxes_np = tf.expand_dims(labels['groundtruth_boxes'].numpy()[idx][0], axis=0).numpy()
    print(boxes_np)
    plot_detections(
      image_np,
      boxes_np,
      np.ones(shape=[labels['groundtruth_boxes'].shape[0]], dtype=np.int32),
      dummy_scores, category_index, plt_show=True)
