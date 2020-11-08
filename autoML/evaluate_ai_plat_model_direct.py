#import googleapiclient
from googleapiclient import discovery
import numpy as np
import socket
import os

MODEL = 'test_model'
PROJECT = 'tissue-defect-ui'

GS_BUCKET_PATH = f'gs://tissue-defect-dev/models/tissue_boundary'
VERSION = 'v10_30_2020_01'

#GS_BUCKET_PATH = f'gs://tissue-defect-tmp/tmp'
#VERSION = 'v3_untrained'
MODEL_DIR_GS = os.path.join(GS_BUCKET_PATH, 'saved_model')

cred_path = r'c:\tmp\work1\Tissue Defect UI-ML Svc Acct.json'
if os.path.exists(cred_path):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = cred_path
else:
    print("Unable to set Credentials env var - can't find cred file {cred_path}")
    raise ValueError

img_t = np.zeros((224, 224,3), dtype=np.uint8)
instances = [{'input_tensor':img_t.tolist()}]

socket.setdefaulttimeout(600)  # set timeout to 10 minutes
service = discovery.build('ml', 'v1', cache_discovery=False, )
model_version_string = 'projects/{}/models/{}/versions/{}'.format(PROJECT, MODEL, VERSION)
print(model_version_string)

response = service.projects().predict(
    name=model_version_string,
    body={'instances': instances}
).execute()

if 'error' in response:
    raise RuntimeError(response['error'])
else:
  print(f'Success.  # keys={response.keys()}')