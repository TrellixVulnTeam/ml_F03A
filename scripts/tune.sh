#!/bin/bash

declare -r PROJECT=$(gcloud config list project --format "value(core.project)")
declare -r BUCKET=gs://vuzii-ml-mlengine
declare -r DATA_PATH=${BUCKET}/data

declare -r JOB_NAME="tune_$(date +%H%M%S)"
declare -r OUTPUT_PATH=${BUCKET}/${JOB_NAME}

declare -r MODEL_NAME=ml_training
declare -r VERSION_NAME=v0
declare -r RUNTIME_VERSION=1.2

echo
echo "Using job id: " $JOB_NAME
# set -v -e

gcloud ml-engine jobs submit training $JOB_NAME \
--job-dir $OUTPUT_PATH \
--runtime-version $RUNTIME_VERSION \
--module-name trainer2.main \
--package-path ./trainer2/ \
--config ./config/hyperparam.yaml \
-- \
--data_dir "${DATA_PATH}/data/train" \
--train_dir "${BUCKET}/tune/${JOB_NAME}" \
--num_batches 2000 \

# Create model
# gcloud ml-engine models create $MODEL_NAME

# Save version
# gcloud ml-engine versions create $VERSION_NAME \
#   --model "$MODEL_NAME" \
#   --origin "${GCS_PATH}/training/model" \
#   --runtime-version=$RUNTIME_VERSION
