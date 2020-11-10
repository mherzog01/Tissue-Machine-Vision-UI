export PYTHONPATH=$CUR_PROJ_DIR/Tissue-Machine-Vision-UI
OUTPUT_DIRECTORY=/tmp/cur_export
if [ -d $OUTPUT_DIRECTORY ]
then
    echo "Deleting $OUTPUT_DIRECTORY"
    rm -rfI $OUTPUT_DIRECTORY
fi
echo "Saving model to $OUTPUT_DIRECTORY"
python $CUR_PROJ_DIR/Tissue-Machine-Vision-UI/obj_det_custom/exporter_main_v2_custom.py \
    --input_type encoded_image_string_tensor \
    --pipeline_config_path $CUR_PROJ_DIR/gcp/tmp/jobs/object_detection_10_30_2020_20_02_25/cfg/pipeline.config \
    --trained_checkpoint_dir $CUR_PROJ_DIR/gcp/tmp/jobs/object_detection_10_30_2020_20_02_25/model \
    --output_directory $OUTPUT_DIRECTORY \
    --selected_detection_keys='num_detections,detection_boxes,detection_classes,detection_masks,detection_scores'
