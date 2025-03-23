#!/bin/bash
set -e

# Default values
PROJECT_ID="estateiqclone"
ARTIFACT_REGISTRY="estateiq-models"
MODEL_PATH="models/estate_price_prediction"
REGION="us-central1"
MLFLOW_TRACKING_URI="sqlite:///mlflow.db"

# Load environment variables if .env exists
if [ -f .env ]; then
    source .env
fi

# Check for required environment variables
if [ -z "${GMAIL_APP_PASSWORD}" ] || [ -z "${NOTIFICATION_EMAIL}" ]; then
    echo "Error: GMAIL_APP_PASSWORD and NOTIFICATION_EMAIL must be set in .env file"
    exit 1
fi

echo "Submitting Cloud Build..."

gcloud builds submit . \
    --config=cloudbuild.yaml \
    --project=${PROJECT_ID} \
    --substitutions=_PROJECT_ID=${PROJECT_ID},_ARTIFACT_REGISTRY=${ARTIFACT_REGISTRY},_MODEL_PATH=${MODEL_PATH},_REGION=${REGION},_MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI},_GMAIL_APP_PASSWORD="${GMAIL_APP_PASSWORD}",_NOTIFICATION_EMAIL="${NOTIFICATION_EMAIL}"

echo "Build submitted!"
echo "Monitor build: https://console.cloud.google.com/cloud-build/builds?project=${PROJECT_ID}"
