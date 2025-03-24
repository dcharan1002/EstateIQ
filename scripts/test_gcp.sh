#!/bin/bash
set -e

# Load environment variables from .env file
if [ -f ".env" ]; then
    source .env
else
    echo "Error: .env file not found. Please run setup_gcp.sh first."
    exit 1
fi

# Set default values if not in environment
PROJECT_ID=${GOOGLE_CLOUD_PROJECT:-estateiqclone}
REGION=${REGION:-us-central1}
ARTIFACT_REGISTRY=${ARTIFACT_REGISTRY:-estateiq-models}
SERVICE_ACCOUNT=${SERVICE_ACCOUNT:-self-522@estateiqclone.iam.gserviceaccount.com}
MODEL_REGISTRY_BASE=${MODEL_REGISTRY_PATH:-models/estate_price_prediction}

# Test GCP connectivity and permissions
echo "Testing GCP configuration..."

# Check authentication
echo "1. Testing authentication..."
gcloud auth list --format="get(account)" || exit 1

# Check project access
echo "2. Testing project access..."
gcloud projects describe $PROJECT_ID || exit 1

# Test service account access
echo "3. Testing service account access..."
gcloud iam service-accounts describe $SERVICE_ACCOUNT || {
    echo "Error: Cannot access service account. Make sure it exists and you have permission."
    exit 1
}

# Test Artifact Registry access
echo "4. Testing Artifact Registry access..."
gcloud artifacts repositories describe $ARTIFACT_REGISTRY \
    --location=$REGION || {
    echo "Error: Cannot access Artifact Registry. Make sure it exists and you have permission."
    exit 1
}

# Test Cloud Storage access
echo "5. Testing Cloud Storage access..."
gcloud storage ls gs://${PROJECT_ID}-models &> /dev/null || {
    echo "Warning: Cannot access storage bucket. It might need to be created."
}

# Check model registry paths
echo "6. Checking model registry paths..."
PATHS=(
    "$MODEL_REGISTRY_BASE/current"
    "$MODEL_REGISTRY_BASE/versions"
)

for path in "${PATHS[@]}"; do
    echo "Checking $path..."
    if gcloud storage ls gs://${PROJECT_ID}-models/$path &> /dev/null; then
        echo "✓ Path exists: $path"
    else
        echo "? Path not found: $path (will be created during setup)"
    fi
done

# Test IAM roles
echo "7. Testing IAM roles..."
REQUIRED_ROLES=(
    "roles/cloudbuild.builds.editor"
    "roles/storage.objectViewer"
    "roles/artifactregistry.writer"
    "roles/run.admin"
)

for role in "${REQUIRED_ROLES[@]}"; do
    echo "Checking $role..."
    if gcloud projects get-iam-policy $PROJECT_ID \
        --format="table(bindings.role,bindings.members)" \
        --flatten="bindings[].members" \
        | grep -q "serviceAccount:$SERVICE_ACCOUNT.*$role"; then
        echo "✓ Role assigned: $role"
    else
        echo "✗ Missing role: $role"
    fi
done

echo "✅ GCP configuration check complete!"
echo "Note: Any missing components will be created by setup_gcp.sh"
