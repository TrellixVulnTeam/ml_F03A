#!/bin/bash

declare -r PROJECT=$(gcloud config list project --format "value(core.project)")
declare -r BUCKET=gs://storage-ml
declare -r DATA_PATH=${BUCKET}/data

declare -r JOB_NAME="run_dr_$(date +%H%M%S)"
declare -r OUTPUT_PATH=${BUCKET}/runs/${JOB_NAME}

echo "Using job id: " $JOB_NAME
set -v -e

gcloud ml-engine jobs submit training $JOB_NAME \
--job-dir $OUTPUT_PATH \
--runtime-version 1.2 \
--module-name trainer2.main \
--package-path trainer2/ \
--config config/config-dr.yaml \
--region us-east1 \
-- \
--data_dir "${DATA_PATH}/train" \
--train_dir "${BUCKET}/ps-top/dr/1-large" \
--manager_type dr \
--num_batches 1000 \
--sync_training True \
--optimizer adam \
--weight_decay 1.0127390883360755e-05 \
--num_epochs_per_decay 0.64437712622319765 \
--learning_rate_decay_factor 0.98779429011183362 \
--learning_rate 0.0003 \
--activation elu \
--batch_size 128 \
--epsilon 0.003166777342546225 \
