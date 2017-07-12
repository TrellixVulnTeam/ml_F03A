#!/bin/bash

declare -r PROJECT=$(gcloud config list project --format "value(core.project)")
declare -r BUCKET=gs://hpc-ml
#declare -r BUCKET=gs://vuzii-ml-mlengine
declare -r DATA_PATH=${BUCKET}/data

declare -r JOB_NAME="run_$(date +%H%M%S)"
declare -r OUTPUT_PATH=${BUCKET}/runs/${JOB_NAME}

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
--train_dir "${BUCKET}/test/ps/gpu=4,adam,bs=64,lr=0.001" \
--optimizer adam \
--manager_type ps \
--learning_rate 0.001 \
--batch_size 64 \
--num_batches 500 \
