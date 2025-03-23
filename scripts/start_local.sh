#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Default values
PORT=8080
PYTHONPATH=$(pwd)

echo -e "${YELLOW}Starting local EstateIQ server${NC}"

# Create artifacts directory if it doesn't exist
mkdir -p artifacts/current

# Copy latest model if exists
if [ -f "artifacts/model_latest.joblib" ]; then
    cp artifacts/model_latest.joblib artifacts/current/model.joblib
else
    echo -e "${RED}No model found in artifacts/model_latest.joblib${NC}"
    echo -e "Please train a model first using: python -m src.model.train"
    exit 1
fi

# Copy metrics if exists
if [ -f "results/validation/validation_latest.json" ]; then
    cp results/validation/validation_latest.json artifacts/current/metrics.json
fi

# Export environment variables
export PYTHONPATH
export MODEL_DIR="$(pwd)/artifacts"
export PORT=$PORT

# Start Flask app
echo -e "\n${YELLOW}Starting Flask application on port ${PORT}...${NC}"
echo -e "Use Ctrl+C to stop the server\n"

# Start the server
FLASK_ENV=development python src/deployment/app.py

echo -e "\n${GREEN}Server stopped${NC}"
