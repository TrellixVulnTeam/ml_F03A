#!/bin/bash

PROJECT_ID=$(gcloud config list project --format "value(core.project)")
BUCKET_NAME=${PROJECT_ID}-mlengine
BUCKET_ADR=gs://${BUCKET_NAME}/data

gsutil cp -r data $BUCKET_ADR/flowers
