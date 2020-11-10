# Source code for locat predict is in 
# C:\Users\mherzo\AppData\Local\Google\Cloud SDK\google-cloud-sdk\lib\third_party\ml_sdk\cloud\ml\prediction\frameworks\tf_prediction_lib.py
#
#MODEL_DIR="$HOME//projects/tissue-defect-ui/gcp/static/models/tissue_boundary/v10_30_2020_01/model/saved_model"
MODEL_DIR=/tmp/cur_export/saved_model
#JSON_REQUEST=./test.json   # Tensor array
JSON_REQUEST=./test_jpg.json
gcloud ai-platform local predict \
    --model-dir $MODEL_DIR \
    --json-request $JSON_REQUEST \
    --verbosity debug \
    --framework tensorflow 2>&1 | sed 's/\\n/\n/g' > log.txt 2>&1
sed 's/  */ /g' log.txt | sed 's/\] *\[/\]\n\[/g' > log_clean.txt