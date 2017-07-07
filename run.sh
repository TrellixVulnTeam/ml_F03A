#!/bin/bash

declare -r PROJECT=$(gcloud config list project --format "value(core.project)")
declare -r BUCKET=gs://vuzii-ml-mlengine
declare -r DATA_PATH=${BUCKET}/data

declare -r JOB_NAME="test_$(date +%H%M%S)"
declare -r OUTPUT_PATH=${BUCKET}/${JOB_NAME}

declare -r MODEL_NAME=imagenet
declare -r VERSION_NAME=v1

echo
echo "Using job id: " $JOB_NAME
set -v -e

gcloud ml-engine jobs submit training $JOB_NAME \
--job-dir $OUTPUT_PATH \
--runtime-version 1.2 \
--module-name trainer2.main \
--package-path trainer2/ \
--config config/config-large.yaml \
-- \
--data_dir "${DATA_PATH}/data/train" \
--train_dir "${BUCKET}/summary/${JOB_NAME}" \
--trace_file "${BUCKET}/summary/trace/trace.json" \
--graph_file "${BUCKET}/summary/${JOB_NAME}/graph.txt" \
