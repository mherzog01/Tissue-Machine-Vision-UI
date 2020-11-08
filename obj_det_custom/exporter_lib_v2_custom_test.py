import unittest

from obj_det_custom import exporter_lib_v2_custom
from object_detection.protos import pipeline_pb2
from object_detection.core import standard_fields as fields
from object_detection.utils import dataset_util

from google.protobuf import text_format
import numpy as np
from PIL import Image
import io
import os
import six

import tensorflow as tf

class ExportInferenceGraphTest(unittest.TestCase):

  def get_dummy_input(self, input_type):
    """Get dummy input for the given input type."""

    if input_type == 'image_tensor':
      return np.zeros((1, 20, 20, 3), dtype=np.uint8)
    if input_type == 'float_image_tensor':
      return np.zeros((1, 20, 20, 3), dtype=np.float32)
    elif input_type == 'encoded_image_string_tensor':
      image = Image.new('RGB', (20, 20))
      byte_io = io.BytesIO()
      image.save(byte_io, 'PNG')
      return [byte_io.getvalue()]
    elif input_type == 'tf_example':
      image_tensor = tf.zeros((20, 20, 3), dtype=tf.uint8)
      encoded_jpeg = tf.image.encode_jpeg(tf.constant(image_tensor)).numpy()
      example = tf.train.Example(
        features=tf.train.Features(
          feature={
            'image/encoded':
              dataset_util.bytes_feature(encoded_jpeg),
            'image/format':
              dataset_util.bytes_feature(six.b('jpeg')),
            'image/source_id':
              dataset_util.bytes_feature(six.b('image_id')),
          })).SerializeToString()
      return [example]

  def load_trained_checkpoint(self, selected_detection_keys=None):

    input_type = 'image_tensor'
    use_default_serving = True

    ckpt_dir = r'c:\tmp\work4\trained_models\tissue_boundary\ckpts\10_30_2020_20_02_25\model'
    tmp_dir = r'c:\tmp\work4\tmp'
    pipeline_config_path = r'C:\Tmp\Work4\trained_models\tissue_boundary\ckpts\10_30_2020_20_02_25\cfg\pipeline.config'

    output_directory = tmp_dir

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(pipeline_config_path, 'r') as f:
      text_format.Merge(f.read(), pipeline_config)

    exporter_lib_v2_custom.export_inference_graph(
      input_type=input_type,
      pipeline_config=pipeline_config,
      trained_checkpoint_dir=ckpt_dir,
      output_directory=output_directory)

    saved_model_path = os.path.join(output_directory, 'saved_model')
    detect_fn = tf.saved_model.load(saved_model_path)
    detect_fn_sig = detect_fn.signatures['serving_default']
    image = tf.constant(self.get_dummy_input(input_type))
    if use_default_serving:
      detections = detect_fn_sig(input_tensor=image)
    else:
      detections = detect_fn(image)

    detection_fields = fields.DetectionResultFields
    for k in detections:
      try:
        cur_shape = detections[k].shape
      except:
        cur_shape = None
    print(k, cur_shape)
    #self.assertEqual(True, True)

  def test_load_trained_checkpoint(self):
    self.load_trained_checkpoint(self)

  def test_load_trained_checkpoint_with_sel(self):
    self.load_trained_checkpoint(self, selected_detection_keys)


if __name__ == '__main__':
  if True:
    ExportInferenceGraphTest.test_load_trained_checkpoint()
    #ExportInferenceGraphTest.test_load_trained_checkpoint_with_sel()
  else:
    unittest.main()
