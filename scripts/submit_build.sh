#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
NC='\033[0m'

# Load environment variables from .env file
if [ -f ".env" ]; then
    source .env
else
    echo -e "${RED}Error: .env file not found. Please run setup_gcp.sh first.${NC}"
    exit 1
fi

# Set defaults from environment or fallback
PROJECT_ID=${GOOGLE_CLOUD_PROJECT:-estateiqclone}
ARTIFACT_REGISTRY=${ARTIFACT_REGISTRY:-estateiq-models}
REGION=${REGION:-us-central1}
MODEL_PATH=${MODEL_REGISTRY_PATH:-models/estate_price_prediction}
MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-sqlite:///mlflow.db}

# Check for required environment variables
if [ -z "${GMAIL_APP_PASSWORD}" ] || [ -z "${GMAIL_USER}" ] || [ -z "${NOTIFICATION_EMAIL}" ]; then
    echo -e "${RED}Error: GMAIL_APP_PASSWORD and NOTIFICATION_EMAIL must be set in .env file${NC}"
    exit 1
fi

echo "Submitting Cloud Build..."

gcloud builds submit . \
    --config=cloudbuild.yaml \
    --project=${PROJECT_ID} \
    --substitutions=_PROJECT_ID=${PROJECT_ID},_ARTIFACT_REGISTRY=${ARTIFACT_REGISTRY},_MODEL_PATH=${MODEL_PATH},_REGION=${REGION},_GMAIL_USER=${GMAIL_USER},_MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI},_GMAIL_APP_PASSWORD="${GMAIL_APP_PASSWORD}",_NOTIFICATION_EMAIL="${NOTIFICATION_EMAIL}"

echo "Build submitted!"
echo "Monitor build: https://console.cloud.google.com/cloud-build/builds?project=${PROJECT_ID}"
