#!/bin/bash
set -e

# Load environment variables from .env file
if [ -f ".env" ]; then
    source .env
else
    echo "Creating .env file from .env.example..."
    cp .env.example .env
    source .env
fi

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
PROJECT_ID=${GOOGLE_CLOUD_PROJECT:-estateiqclone}
REGION=${REGION:-us-central1}
ARTIFACT_REGISTRY=${ARTIFACT_REGISTRY:-estateiq-models}
SERVICE_ACCOUNT=${SERVICE_ACCOUNT:-self-522@estateiqclone.iam.gserviceaccount.com}

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
if ! gcloud artifacts repositories describe $ARTIFACT_REGISTRY --location=$REGION &> /dev/null; then
    gcloud artifacts repositories create $ARTIFACT_REGISTRY \
        --repository-format=docker \
        --location=$REGION \
        --description="Repository for EstateIQ ML models"
    echo "Artifact Registry repository created."
else
    echo "Artifact Registry repository already exists, skipping creation."
fi

# Create service account key if not exists
if [ ! -f "key.json" ]; then
    echo "Creating service account key..."
    gcloud iam service-accounts keys create key.json \
        --iam-account=$SERVICE_ACCOUNT
            
    # Update GOOGLE_APPLICATION_CREDENTIALS in .env
    sed -i.bak "s|GOOGLE_APPLICATION_CREDENTIALS=.*|GOOGLE_APPLICATION_CREDENTIALS=$(pwd)/key.json|" .env
    rm .env.bak
fi

# Grant necessary permissions
echo "Granting permissions to service account..."
for role in "roles/cloudbuild.builds.editor" "roles/storage.objectViewer" "roles/artifactregistry.writer" "roles/run.admin"; do
    if ! gcloud projects get-iam-policy $PROJECT_ID \
        --flatten="bindings[].members" \
        --format='table(bindings.members)' \
        --filter="bindings.role:$role AND bindings.members:'serviceAccount:$SERVICE_ACCOUNT'" | \
        grep -q "serviceAccount:$SERVICE_ACCOUNT"; then
        
        echo "Adding $role..."
        gcloud projects add-iam-policy-binding $PROJECT_ID \
            --member="serviceAccount:$SERVICE_ACCOUNT" \
            --role="$role"
    else
        echo "Role $role already assigned, skipping."
    fi
done

# Create storage buckets if they don't exist
echo "Setting up Cloud Storage buckets..."
if ! gcloud storage ls gs://${PROJECT_ID}-models &> /dev/null; then
    gcloud storage buckets create gs://${PROJECT_ID}-models \
        --location=$REGION \
        --uniform-bucket-level-access
    echo "Created model storage bucket."
else
    echo "Model storage bucket already exists."
fi

# Create model registry paths
MODEL_REGISTRY_BASE=${MODEL_REGISTRY_PATH:-models/estate_price_prediction}
for path in "$MODEL_REGISTRY_BASE/current" "$MODEL_REGISTRY_BASE/versions"; do
    echo "Creating $path directory..."
    # Using gcloud storage instead of gsutil
    if ! gcloud storage ls gs://${PROJECT_ID}-models/$path &> /dev/null; then
        # Create an empty file to establish the directory
        touch temp_file
        gcloud storage cp temp_file gs://${PROJECT_ID}-models/$path/.placeholder
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
echo "2. Environment file (.env) has been created/updated with the correct values"
echo "3. Push your changes to trigger the pipeline!"
