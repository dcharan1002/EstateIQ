#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_step() {
    echo -e "\n${YELLOW}=== $1 ===${NC}\n"
}

check_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $1 successful${NC}"
    else
        echo -e "${RED}✗ $1 failed${NC}"
        exit 1
    fi
}

# Check required environment variables
print_step "Checking environment variables"
required_vars=("GOOGLE_APPLICATION_CREDENTIALS" "GMAIL_APP_PASSWORD")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo -e "${RED}Error: $var is not set${NC}"
        exit 1
    fi
done
check_status "Environment check"

# Clean up any existing containers and artifacts
print_step "Cleaning up previous runs"
docker compose down -v
rm -rf artifacts/* results/* mlruns/* mlflow.db
check_status "Cleanup"

# Pull data using DVC
print_step "Pulling data from DVC"
dvc pull
check_status "DVC pull"

# Build containers
print_step "Building Docker containers"
docker compose build
check_status "Docker build"

# Run tests
print_step "Running tests"
docker compose run --rm model-training python -m pytest tests/
check_status "Tests"

# Train model
print_step "Training model"
docker compose run --rm model-training python src/model/train.py
check_status "Model training"

# Check if model artifacts exist
print_step "Verifying model artifacts"
if [ ! -f "artifacts/model_latest.joblib" ]; then
    echo -e "${RED}Error: Model artifact not found${NC}"
    exit 1
fi
check_status "Artifact verification"

# Check MLflow tracking
print_step "Checking MLflow records"
if [ ! -f "mlflow.db" ]; then
    echo -e "${RED}Error: MLflow database not found${NC}"
    exit 1
fi
check_status "MLflow verification"

# Start model server
print_step "Starting model server"
docker compose up -d model-server
check_status "Model server startup"

# Wait for server to be ready
echo "Waiting for server to be ready..."
sleep 10

# Test prediction endpoint
print_step "Testing prediction endpoint"
curl -X POST -H "Content-Type: application/json" \
     -d '{"features": [0.5, 0.5, 0.5, 0.5]}' \
     http://localhost:8000/predict
check_status "Prediction test"

# Check results directory
print_step "Checking analysis results"
if [ ! -d "results" ] || [ -z "$(ls -A results)" ]; then
    echo -e "${RED}Error: No analysis results found${NC}"
    exit 1
fi
check_status "Results verification"

# Stop containers
print_step "Cleaning up"
docker compose down
check_status "Cleanup"

echo -e "\n${GREEN}Pipeline test completed successfully!${NC}"
echo -e "\nArtifacts generated:"
echo "- Model: artifacts/model_latest.joblib"
echo "- Results: results/"
echo "- MLflow DB: mlflow.db"
echo "- MLflow runs: mlruns/"
