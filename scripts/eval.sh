#!/bin/bash

declare -r PROJECT=$(gcloud config list project --format "value(core.project)")
declare -r BUCKET=gs://hpc-ml
declare -r DATA_PATH=${BUCKET}/data

declare -r JOB_NAME="eval_$(date +%H%M%S)"
declare -r OUTPUT_PATH=${BUCKET}/${JOB_NAME}

echo
echo "Using job id: " $JOB_NAME
set -v -e

gcloud ml-engine jobs submit training $JOB_NAME \
--job-dir $OUTPUT_PATH \
--runtime-version 1.2 \
--module-name trainer2.main \
--config config/config-eval.yaml \
--package-path trainer2/ \
-- \
--run_training False \
--data_dir "${DATA_PATH}/validation" \
--train_dir "${BUCKET}/test/ps/gpu=2,async" \
--manager_type local \
--run_training False \
--num_batches 500 \
--batch_size 64 \
--debug_level 4 \
