# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 16:43:59 2020

@author: MHerzo
"""


from google.cloud import automl

project_id = "tissue-defect-ui"
model_id = "IOD272146720060604416"


# Get eval ID

client = automl.AutoMlClient()
# Get the full path of the model.
model_full_id = client.model_path(project_id, "us-central1", model_id)

print("List of model evaluations:")
for evaluation in client.list_model_evaluations(model_full_id, ""):
    print("Model evaluation name: {}".format(evaluation.name))
    print(
        "Model annotation spec id: {}".format(
            evaluation.annotation_spec_id
        )
    )
    print("Create Time:")
    print("\tseconds: {}".format(evaluation.create_time.seconds))
    print("\tnanos: {}".format(evaluation.create_time.nanos / 1e9))
    print(
        "Evaluation example count: {}".format(
            evaluation.evaluated_example_count
        )
    )
    print(
        "Object detection model evaluation metrics: {}\n\n".format(
            evaluation.image_object_detection_evaluation_metrics
        )
    )

# -----------------------------------------------
# Evaluate

model_evaluation_id = "YOUR_MODEL_EVALUATION_ID"

client = automl.AutoMlClient()
# Get the full path of the model evaluation.
model_evaluation_full_id = client.model_evaluation_path(
    project_id, "us-central1", model_id, model_evaluation_id
)

# Get complete detail of the model evaluation.
response = client.get_model_evaluation(model_evaluation_full_id)

print("Model evaluation name: {}".format(response.name))
print("Model annotation spec id: {}".format(response.annotation_spec_id))
print("Create Time:")
print("\tseconds: {}".format(response.create_time.seconds))
print("\tnanos: {}".format(response.create_time.nanos / 1e9))
print(
    "Evaluation example count: {}".format(response.evaluated_example_count)
)
print(
    "Object detection model evaluation metrics: {}".format(
        response.image_object_detection_evaluation_metrics
    )
)