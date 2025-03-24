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
ARTIFACT_REGISTRY=${ARTIFACT_REGISTRY:-estateiq-models}
MODEL_PATH=${MODEL_REGISTRY_PATH:-models/estate_price_prediction}
SERVICE_ACCOUNT=${SERVICE_ACCOUNT:-self-522@estateiqclone.iam.gserviceaccount.com}
SERVICE_NAME="estateiq-prediction"
MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-sqlite:///mlflow.db}

echo -e "${YELLOW}Updating GitHub workflow files with environment variables...${NC}"

# Function to update environment variables in workflow files
update_workflow_file() {
    local file=$1
    local temp_file="${file}.tmp"
    
    # Create a temporary file
    cp "$file" "$temp_file"
    
    # Update environment variables
    sed -i.bak "s/PROJECT_ID: .*/PROJECT_ID: $PROJECT_ID/" "$temp_file"
    sed -i.bak "s/REGION: .*/REGION: $REGION/" "$temp_file"
    sed -i.bak "s/ARTIFACT_REGISTRY: .*/ARTIFACT_REGISTRY: $ARTIFACT_REGISTRY/" "$temp_file"
    sed -i.bak "s|MODEL_PATH: .*|MODEL_PATH: $MODEL_PATH|" "$temp_file"
    sed -i.bak "s/SA_EMAIL: .*/SA_EMAIL: $SERVICE_ACCOUNT/" "$temp_file"
    
    # Additional variables specific to training workflow
    if [[ $file == *"model-training.yml" ]]; then
        sed -i.bak "s|MLFLOW_TRACKING_URI: .*|MLFLOW_TRACKING_URI: $MLFLOW_TRACKING_URI|" "$temp_file"
    fi
    
    # Additional variables specific to deployment workflow
    if [[ $file == *"model-deployment.yml" ]]; then
        sed -i.bak "s/SERVICE_NAME: .*/SERVICE_NAME: $SERVICE_NAME/" "$temp_file"
    fi
    
    # Remove backup files
    rm -f "$temp_file.bak"
    
    # Replace original file with updated one
    mv "$temp_file" "$file"
    
    echo -e "${GREEN}Updated $file${NC}"
}

# Update both workflow files
update_workflow_file ".github/workflows/model-training.yml"
update_workflow_file ".github/workflows/model-deployment.yml"

echo -e "\n${GREEN}Successfully updated environment variables in workflow files!${NC}"
echo -e "Changes made:"
echo -e "  PROJECT_ID: $PROJECT_ID"
echo -e "  REGION: $REGION"
echo -e "  ARTIFACT_REGISTRY: $ARTIFACT_REGISTRY"
echo -e "  MODEL_PATH: $MODEL_PATH"
echo -e "  SERVICE_ACCOUNT: $SERVICE_ACCOUNT"
echo -e "  MLFLOW_TRACKING_URI: $MLFLOW_TRACKING_URI"
echo -e "  SERVICE_NAME: $SERVICE_NAME"
