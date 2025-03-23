#!/bin/bash
set -e

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud CLI is not installed. Please install it first."
    exit 1
fi

# Check if already authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="get(account)" | grep -q .; then
    echo "Not authenticated. Please run: gcloud auth login"
    exit 1
fi

# Configure project
export PROJECT_ID=estateiqclone
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "Enabling required GCP APIs..."
gcloud services enable \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    run.googleapis.com \
    storage.googleapis.com

# Set up Artifact Registry
echo "Setting up Artifact Registry..."
if ! gcloud artifacts repositories describe estateiq-models --location=us-central1 &> /dev/null; then
    gcloud artifacts repositories create estateiq-models \
        --repository-format=docker \
        --location=us-central1 \
        --description="Repository for EstateIQ ML models"
    echo "Artifact Registry repository created."
else
    echo "Artifact Registry repository already exists, skipping creation."
fi

# Create service account key if not exists
if [ ! -f "key.json" ]; then
    echo "Creating service account key..."
    gcloud iam service-accounts keys create key.json \
        --iam-account=self-522@estateiqclone.iam.gserviceaccount.com
fi

# Grant necessary permissions
echo "Granting permissions to service account..."
for role in "roles/cloudbuild.builds.editor" "roles/storage.objectViewer" "roles/artifactregistry.writer" "roles/run.admin"; do
    if ! gcloud projects get-iam-policy $PROJECT_ID \
        --flatten="bindings[].members" \
        --format='table(bindings.members)' \
        --filter="bindings.role:$role AND bindings.members:'serviceAccount:self-522@estateiqclone.iam.gserviceaccount.com'" | \
        grep -q "serviceAccount:self-522@estateiqclone.iam.gserviceaccount.com"; then
        
        echo "Adding $role..."
        gcloud projects add-iam-policy-binding $PROJECT_ID \
            --member="serviceAccount:self-522@estateiqclone.iam.gserviceaccount.com" \
            --role="$role"
    else
        echo "Role $role already assigned, skipping."
    fi
done

# Create storage buckets if they don't exist
echo "Setting up Cloud Storage buckets..."
if ! gcloud storage ls gs://estateiq-models &> /dev/null; then
    gcloud storage buckets create gs://estateiq-models \
        --location=us-central1 \
        --uniform-bucket-level-access
    echo "Created model storage bucket."
else
    echo "Model storage bucket already exists."
fi

# Create model registry paths
for path in "models/estate_price_prediction/current" "models/estate_price_prediction/versions"; do
    echo "Creating $path directory..."
    # Using gcloud storage instead of gsutil
    if ! gcloud storage ls gs://estateiq-models/$path &> /dev/null; then
        # Create an empty file to establish the directory
        touch temp_file
        gcloud storage cp temp_file gs://estateiq-models/$path/.placeholder
        rm temp_file
        echo "Created $path directory."
    else
        echo "$path directory already exists."
    fi
done

echo "✅ GCP setup complete!"
echo ""
echo "Next steps:"
echo "1. Add the contents of key.json as a GitHub secret named 'GCP_SA_KEY'"
echo "2. Copy .env.example to .env and update values"
echo "3. Push your changes to trigger the pipeline!"
