#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Load environment variables from .env file
if [ -f ".env" ]; then
    source .env
else
    echo -e "${RED}Error: .env file not found. Please run setup_gcp.sh first.${NC}"
    exit 1
fi

# Set defaults from environment or fallback values
PROJECT_ID=${GOOGLE_CLOUD_PROJECT:-estateiqclone}
REGION=${REGION:-us-central1}
SERVICE_NAME="estateiq-prediction"
ARTIFACT_REGISTRY=${ARTIFACT_REGISTRY:-estateiq-models}
MODEL_PATH=${MODEL_REGISTRY_PATH:-models/estate_price_prediction}
SERVICE_ACCOUNT=${SERVICE_ACCOUNT:-self-522@estateiqclone.iam.gserviceaccount.com}

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --project-id) PROJECT_ID="$2"; shift ;;
        --region) REGION="$2"; shift ;;
        --service-name) SERVICE_NAME="$2"; shift ;;
        --artifact-registry) ARTIFACT_REGISTRY="$2"; shift ;;
        --model-path) MODEL_PATH="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo -e "${YELLOW}Deploying service to Cloud Run...${NC}"
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo "Service: $SERVICE_NAME"
echo "Artifact Registry: $ARTIFACT_REGISTRY"
echo "Model Path: $MODEL_PATH"

# Deploy to Cloud Run with environment variables
gcloud run deploy ${SERVICE_NAME} \
    --image ${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REGISTRY}/prediction-service:latest \
    --region ${REGION} \
    --platform managed \
    --allow-unauthenticated \
    --project ${PROJECT_ID} \
    --service-account ${SERVICE_ACCOUNT} \
    --set-env-vars="MODEL_ARTIFACTS_BUCKET=${ARTIFACT_REGISTRY},MODEL_REGISTRY_PATH=${MODEL_PATH}" \
    --min-instances=1 \
    --timeout=300s

echo -e "\n${GREEN}Service deployed successfully!${NC}"
echo -e "Use this command to get the service URL:"
echo -e "${YELLOW}gcloud run services describe ${SERVICE_NAME} --platform managed --region ${REGION} --format='value(status.url)'${NC}"
