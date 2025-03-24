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

# Set defaults from environment or fallback
PORT=${PORT:-8080}
MODEL_DIR=${MODEL_DIR:-$(pwd)/artifacts}
PYTHONPATH=$(pwd)
APP_PID=""

# Cleanup function
cleanup() {
    if [ ! -z "$APP_PID" ]; then
        echo -e "\n${YELLOW}Cleaning up...${NC}"
        kill $APP_PID 2>/dev/null || true
    fi
}

# Set trap for cleanup
trap cleanup EXIT INT TERM

echo -e "${YELLOW}Setting up local test environment${NC}"

# Create artifacts directory if it doesn't exist
mkdir -p artifacts/current

# Copy latest model if exists
if [ -f "artifacts/model_latest.joblib" ]; then
    cp artifacts/model_latest.joblib artifacts/current/model.joblib
else
    echo -e "${RED}No model found in artifacts/model_latest.joblib${NC}"
    exit 1
fi

# Copy metrics if exists
if [ -f "results/validation/validation_latest.json" ]; then
    cp results/validation/validation_latest.json artifacts/current/metrics.json
fi

# Export environment variables
export PYTHONPATH
export MODEL_DIR
export PORT

# Set validation thresholds from environment
export VALIDATION_R2_THRESHOLD=${VALIDATION_R2_THRESHOLD:-0.8}
export VALIDATION_RMSE_THRESHOLD=${VALIDATION_RMSE_THRESHOLD:-1.0}
export VALIDATION_MAE_THRESHOLD=${VALIDATION_MAE_THRESHOLD:-0.8}

# Create test data with minimal features
echo -e "\n${YELLOW}Creating test data with minimal features...${NC}"
cat > test_data.json << 'END_JSON'
{
    "GROSS_AREA": 2500,
    "LIVING_AREA": 2000,
    "LAND_SF": 5000,
    "YR_BUILT": 1990,
    "BED_RMS": 3,
    "FULL_BTH": 2,
    "HLF_BTH": 1
}
END_JSON

# Start Flask app in the background
echo -e "\n${YELLOW}Starting Flask application...${NC}"
python src/deployment/app.py &
APP_PID=$!

# Wait for app to start
echo -e "Waiting for service to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:$PORT/health > /dev/null; then
        echo -e "${GREEN}Service is ready!${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}Timeout waiting for service to start${NC}"
        exit 1
    fi
    sleep 1
done

# Test endpoints
echo -e "\n${YELLOW}Testing health endpoint...${NC}"
curl -s http://localhost:$PORT/health | python -m json.tool

echo -e "\n${YELLOW}Testing prediction endpoint with minimal features...${NC}"
curl -s -X POST \
    -H "Content-Type: application/json" \
    -d @test_data.json \
    http://localhost:$PORT/predict | python -m json.tool

# Create another test case with different features
echo -e "\n${YELLOW}Testing with different feature set...${NC}"
cat > test_data2.json << 'END_JSON'
{
    "GROSS_AREA": 3000,
    "LIVING_AREA": 2500,
    "LAND_SF": 6000,
    "YR_BUILT": 1850,
    "BED_RMS": 10,
    "STRUCTURE_CLASS_C - BRICK/CONCR": 1,
    "INT_COND_A - AVERAGE": 0
}
END_JSON

echo -e "\n${YELLOW}Testing prediction endpoint with alternative features...${NC}"
curl -s -X POST \
    -H "Content-Type: application/json" \
    -d @test_data2.json \
    http://localhost:$PORT/predict | python -m json.tool

# Cleanup temporary files
rm -f test_data.json test_data2.json

echo -e "\n${GREEN}Local testing complete!${NC}"
echo -e "You can now deploy to Cloud Run using: ./scripts/start_service.sh"
