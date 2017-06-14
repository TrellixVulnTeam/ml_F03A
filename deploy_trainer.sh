#!/bin/bash

JOB_NAME=flowers_12

PROJECT_ID=$(gcloud config list project --format "value(core.project)")
BUCKET_NAME=${PROJECT_ID}-mlengine
REGION=us-central1

TRAIN_DATA=gs://$BUCKET_NAME/data/flowers/data

OUTPUT_PATH=gs://$BUCKET_NAME/$JOB_NAME
SUMMARY_PATH=gs://$BUCKET_NAME/summary/$JOB_NAME/

gcloud ml-engine jobs submit training $JOB_NAME \
--job-dir $OUTPUT_PATH \
--module-name trainer.task \
--package-path trainer/ \
-- \
--train_dir $SUMMARY_PATH \
--data_dir $TRAIN_DATA \
