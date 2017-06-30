#!/bin/bash

declare -r PROJECT=$(gcloud config list project --format "value(core.project)")
declare -r BUCKET=gs://vuzii-ml-mlengine
declare -r DATA_PATH=${BUCKET}/data

declare -r JOB_NAME=test_4
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
--module-name data.build_data \
--package-path data/ \
-- \
--data_dir "${DATA_PATH}/raw_data" \
--output_dir "${DATA_PATH}/data" \
--download_list "${DATA_PATH}/txt_util/sets_to_download.txt" \
--labels_file "${DATA_PATH}/txt_util/synset.txt" \
--labels_dir "${DATA_PATH}/data/Annotation" \
--imagenet_metadata_file "${DATA_PATH}/txt_util/imagenet_metadata.txt" \
--bounding_box_file "${DATA_PATH}/txt_util/bounding_boxes.csv" \
