#!/bin/bash
set -e

# Test GCP connectivity and permissions
echo "Testing GCP configuration..."

# Check authentication
echo "1. Testing authentication..."
gcloud auth list --format="get(account)" || exit 1

# Check project access
echo "2. Testing project access..."
gcloud projects describe estateiqclone || exit 1

# Test service account access
echo "3. Testing service account access..."
gcloud iam service-accounts describe self-522@estateiqclone.iam.gserviceaccount.com || {
    echo "Error: Cannot access service account. Make sure it exists and you have permission."
    exit 1
}

# Test Artifact Registry access
echo "4. Testing Artifact Registry access..."
gcloud artifacts repositories describe estateiq-models \
    --location=us-central1 || {
    echo "Error: Cannot access Artifact Registry. Make sure it exists and you have permission."
    exit 1
}

# Test Cloud Storage access
echo "5. Testing Cloud Storage access..."
gcloud storage ls gs://estateiq-models &> /dev/null || {
    echo "Warning: Cannot access storage bucket. It might need to be created."
}

# Check model registry paths
echo "6. Checking model registry paths..."
PATHS=(
    "models/estate_price_prediction/current"
    "models/estate_price_prediction/versions"
)

for path in "${PATHS[@]}"; do
    echo "Checking $path..."
    if gcloud storage ls gs://estateiq-models/$path &> /dev/null; then
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
    if gcloud projects get-iam-policy estateiqclone \
        --format="table(bindings.role,bindings.members)" \
        --flatten="bindings[].members" \
        | grep -q "serviceAccount:self-522@estateiqclone.iam.gserviceaccount.com.*$role"; then
        echo "✓ Role assigned: $role"
    else
        echo "✗ Missing role: $role"
    fi
done

echo "✅ GCP configuration check complete!"
echo "Note: Any missing components will be created by setup_gcp.sh"
