#!/bin/bash

declare -r PROJECT=$(gcloud config list project --format "value(core.project)")
declare -r BUCKET=gs://storage-ml
declare -r DATA_PATH=${BUCKET}/data

declare -r JOB_NAME="run_ps_a_$(date +%H%M%S)"
declare -r OUTPUT_PATH=${BUCKET}/runs/${JOB_NAME}

echo "Using job id: " $JOB_NAME
set -v -e

gcloud ml-engine jobs submit training $JOB_NAME \
--job-dir $OUTPUT_PATH \
--runtime-version 1.2 \
--module-name trainer2.main \
--package-path trainer2/ \
--config config/config-ps-a.yaml \
--region us-east1 \
-- \
--data_dir "${DATA_PATH}/train" \
--train_dir "${BUCKET}/ps-top/dr/4-2" \
--manager_type ps \
--num_batches 1000 \
--sync_training False \
--optimizer momentum \
--weight_decay 0.00043779348920769406 \
--num_epochs_per_decay 0.51748904584121136 \
--learning_rate_decay_factor 0.98775308754993685 \
--learning_rate 0.009983 \
--momentum 0.87848350610230053 \
--activation relu \
--batch_size 256 \
