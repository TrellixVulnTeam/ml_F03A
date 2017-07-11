#!/bin/bash

declare -r PROJECT=$(gcloud config list project --format "value(core.project)")
declare -r BUCKET=gs://hpc-ml
#declare -r BUCKET=gs://vuzii-ml-mlengine
declare -r DATA_PATH=${BUCKET}/data

declare -r JOB_NAME="run_$(date +%H%M%S)"
declare -r OUTPUT_PATH=${BUCKET}/${JOB_NAME}

declare -r MODEL_NAME=deep_learn
declare -r VERSION_NAME=v1.0

echo
echo "Using job id: " $JOB_NAME
set -v -e

gcloud ml-engine jobs submit training $JOB_NAME \
--job-dir $OUTPUT_PATH \
--runtime-version 1.2 \
--module-name trainer2.main \
--package-path trainer2/ \
--config config/config-large.yaml \
--region us-east1 \
-- \
--data_dir "${DATA_PATH}/train" \
--train_dir "${BUCKET}/result/ps/gpu=4" \
--variable_update "parameter_server" \
--cross_replica_sync True \
--staged_vars True \
--sync_on_finish True \
--learning_rate 0.005 \
--batch_size 64 \
--num_batches 8000

gcloud ml-engine models create "$MODEL_NAME" \
--regions us-east1 \
