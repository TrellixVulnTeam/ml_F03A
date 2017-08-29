#!/bin/bash

declare -r PROJECT=$(gcloud config list project --format "value(core.project)")
declare -r BUCKET=gs://storage-ml
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
--region us-east1 \
-- \
--run_training False \
--data_dir "${DATA_PATH}/validation" \
--train_dir "${BUCKET}/topology/ps/4xsingle_gpu,2ps" \
--manager_type local \
--optimizer momentum \
--num_batches 500 \
--batch_size 256 \
--debug_level 4 \
