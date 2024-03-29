#!/bin/bash

declare -r PROJECT=$(gcloud config list project --format "value(core.project)")
declare -r BUCKET=gs://vuzii-ml-mlengine
declare -r DATA_PATH=${BUCKET}/data

declare -r JOB_NAME=bench_test_2
declare -r OUTPUT_PATH=${BUCKET}/${JOB_NAME}

declare -r MODEL_NAME=imagenet
declare -r VERSION_NAME=v1

echo
echo "Using job id: " $JOB_NAME
set -v -e

# gsutil cp -r data/txt_util $DATA_PATH
# gsutil cp -r data/*.py $DATA_PATH

gcloud ml-engine jobs submit training $JOB_NAME \
--job-dir $OUTPUT_PATH \
--runtime-version 1.2 \
--module-name trainer.task \
--package-path trainer/ \
--scale-tier BASIC_GPU \
-- \
--train_dir "${BUCKET}/summary/${JOB_NAME}" \
