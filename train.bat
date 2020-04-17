@echo off
set PIPELINE_CONFIG_PATH=C:\Users\mherzo\Documents\GitHub\tf-models\app\models\cur\ssd_mobilenet_v1_ui.config
set MODEL_DIR=C:\Users\mherzo\Documents\GitHub\tf-models\app\models\cur\ssd_mobilenet_v1_coco_2018_01_28
rem cur_checkpoint=7442482
rem set NUM_TRAIN_STEPS=7443000
set NUM_TRAIN_STEPS=500
set SAMPLE_1_OF_N_EVAL_EXAMPLES=1
set TF_RESEARCH=C:\Users\mherzo\Documents\GitHub\tf-models\models\research
set PYTHONPATH=%TF_RESEARCH%;%TF_RESEARCH%\slim
cd %TF_RESEARCH%
python object_detection/model_main.py --pipeline_config_path=%PIPELINE_CONFIG_PATH% --model_dir=%MODEL_DIR% --num_train_steps=%NUM_TRAIN_STEPS% --sample_1_of_n_eval_examples=%SAMPLE_1_OF_N_EVAL_EXAMPLES% --alsologtostderr
