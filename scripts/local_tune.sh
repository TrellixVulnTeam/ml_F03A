#!/bin/bash

declare -r PROJECT=$(gcloud config list project --format "value(core.project)")
declare -r BUCKET=gs://hpc-ml
declare -r DATA_PATH=${BUCKET}/data

declare -r JOB_NAME="tune_$(date +%H%M%S)"
declare -r OUTPUT_PATH=${BUCKET}/tune/${JOB_NAME}

declare -r MODEL_NAME=ml_training
declare -r VERSION_NAME=v0
declare -r RUNTIME_VERSION=1.2

echo "Using job id: " $JOB_NAME
set -v -e

gcloud ml-engine jobs submit training $JOB_NAME \
--job-dir $OUTPUT_PATH \
--runtime-version $RUNTIME_VERSION \
--module-name trainer2.main \
--package-path ./trainer2/ \
--config ./config/local_hp.yaml \
--region us-east1 \
-- \
--data_dir "${DATA_PATH}/train" \
--train_dir "${BUCKET}/hp_tuning/local/adam" \
--optimizer adam \
--manager_type local \
--num_batches 1000 \
