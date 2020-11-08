import os
import unittest
from obj_det_custom import exporter_lib_v2_custom
from object_detection.protos import pipeline_pb2
from object_detection.core import standard_fields as fields
from google.protobuf import text_format

import tensorflow as tf

class ExportInferenceGraphTest(unittest.TestCase):
    def test_load_trained_checkpoint(self):

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
        print(detections.keys())
        #self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
