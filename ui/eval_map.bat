@echo off
set TF_RESEARCH=C:\Users\mherzo\Documents\GitHub\tf-models\models\research
set PYTHONPATH=C:\Users\mherzo\Documents\GitHub\tf-models\app\ui;%TF_RESEARCH%;%TF_RESEARCH%\slim;C:\Users\mherzo\Documents\GitHub\labelme
cd %TF_RESEARCH%
rem python C:\Users\mherzo\Documents\GitHub\tf-models\app\ui\evaluate_automl_multiproc_mapped.py
python C:\Users\mherzo\Documents\GitHub\tf-models\app\ui\eval_with_cv2_tracking.py
