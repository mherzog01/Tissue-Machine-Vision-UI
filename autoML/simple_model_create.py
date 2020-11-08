from object_detection.builders import model_builder
from object_detection.utils import config_util

pipeline_config = 'c:/tmp/work4/object_detection/configs/tf2/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.config'

configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']

detection_model = model_builder.build(model_config,is_training=True,add_summaries=True)