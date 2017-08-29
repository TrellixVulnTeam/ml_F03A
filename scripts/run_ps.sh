#!/bin/bash

declare -r PROJECT=$(gcloud config list project --format "value(core.project)")
declare -r BUCKET=gs://storage-ml
declare -r DATA_PATH=${BUCKET}/data

declare -r JOB_NAME="run_ps_$(date +%H%M%S)"
declare -r OUTPUT_PATH=${BUCKET}/runs/${JOB_NAME}

echo "Using job id: " $JOB_NAME
set -v -e

gcloud ml-engine jobs submit training $JOB_NAME \
--job-dir $OUTPUT_PATH \
--runtime-version 1.2 \
--module-name trainer2.main \
--package-path trainer2/ \
--config config/config-ps.yaml \
--region us-east1 \
-- \
--data_dir "${DATA_PATH}/train" \
--train_dir "${BUCKET}/tune2/ps/run=1" \
--manager_type ps \
--num_batches 2000 \
--sync_training True \
--optimizer adam \
--weight_decay 0.00050555198684637038 \
--num_epochs_per_decay 0.9410562708324588 \
--learning_rate_decay_factor 0.9796150486288504 \
--learning_rate 0.001132 \
--activation relu \
--batch_size 512 \
--epsilon 0.007338138895 \
