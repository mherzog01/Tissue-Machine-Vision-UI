@echo off
set TF_RESEARCH=C:\Users\mherzo\Documents\GitHub\tf-models\models\research
set PYTHONPATH=%TF_RESEARCH%;%TF_RESEARCH%\slim
cd %TF_RESEARCH%
python C:\Users\mherzo\Documents\GitHub\tf-models\app\evaluate_automl_multiproc.py
