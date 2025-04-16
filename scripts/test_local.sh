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
export VALIDATION_R2_THRESHOLD=${VALIDATION_R2_THRESHOLD:-0.6}
export VALIDATION_RMSE_THRESHOLD=${VALIDATION_RMSE_THRESHOLD:-50000}
export VALIDATION_MAE_THRESHOLD=${VALIDATION_MAE_THRESHOLD:-35000}

# Create typical single-family home test data
echo -e "\n${YELLOW}Creating typical single-family home test data...${NC}"
cat > test_data.json << 'END_JSON'
{
    "GROSS_AREA": 2500.0,
    "LIVING_AREA": 2000.0,
    "LAND_SF": 5000.0,
    "YR_BUILT": 1990,
    "YR_REMODEL": 2015,
    "BED_RMS": 3.0,
    "FULL_BTH": 2.0,
    "HLF_BTH": 1.0,
    "NUM_PARKING": 2,
    "FIREPLACES": 1,
    "KITCHENS": 1,
    "TT_RMS": 8,
    "ZIP_CODE": "02108",
    "STRUCTURE_CLASS_C - BRICK/CONCR": 1,
    "STRUCTURE_CLASS_D - WOOD/FRAME": 0,
    "STRUCTURE_CLASS_B - REINF CONCR": 0,
    "INT_COND_E - EXCELLENT": 0,
    "INT_COND_G - GOOD": 1,
    "INT_COND_A - AVERAGE": 0,
    "INT_COND_F - FAIR": 0,
    "INT_COND_P - POOR": 0,
    "OVERALL_COND_E - EXCELLENT": 0,
    "OVERALL_COND_VG - VERY GOOD": 1,
    "OVERALL_COND_G - GOOD": 0,
    "OVERALL_COND_A - AVERAGE": 0,
    "OVERALL_COND_F - FAIR": 0,
    "OVERALL_COND_P - POOR": 0,
    "KITCHEN_STYLE2_M - MODERN": 1,
    "KITCHEN_STYLE2_L - LUXURY": 0,
    "KITCHEN_STYLE2_S - SEMI-MODERN": 0,
    "KITCHEN_TYPE_F - FULL EAT IN": 1,
    "AC_TYPE_C - CENTRAL AC": 1,
    "AC_TYPE_D - DUCTLESS AC": 0,
    "HEAT_TYPE_F - FORCED HOT AIR": 1,
    "HEAT_TYPE_W - HT WATER/STEAM": 0,
    "PROP_VIEW_E - EXCELLENT": 0,
    "PROP_VIEW_G - GOOD": 1,
    "CORNER_UNIT_Y - YES": 0,
    "ORIENTATION_E - END": 0,
    "ORIENTATION_F - FRONT/STREET": 1,
    "EXT_COND_E - EXCELLENT": 0,
    "EXT_COND_G - GOOD": 1,
    "ROOF_COVER_S - SLATE": 0,
    "ROOF_COVER_A - ASPHALT SHINGL": 1
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

# Create luxury historic home test data
echo -e "\n${YELLOW}Creating luxury historic home test data...${NC}"
cat > test_data2.json << 'END_JSON'
{
    "GROSS_AREA": 3000.0,
    "LIVING_AREA": 2500.0,
    "LAND_SF": 6000.0,
    "YR_BUILT": 1850,
    "YR_REMODEL": 2020,
    "BED_RMS": 5.0,
    "FULL_BTH": 3.0,
    "HLF_BTH": 1.0,
    "NUM_PARKING": 3,
    "FIREPLACES": 2,
    "KITCHENS": 2,
    "TT_RMS": 12,
    "ZIP_CODE": "02108",
    "STRUCTURE_CLASS_C - BRICK/CONCR": 1,
    "STRUCTURE_CLASS_D - WOOD/FRAME": 0,
    "STRUCTURE_CLASS_B - REINF CONCR": 0,
    "INT_COND_E - EXCELLENT": 1,
    "INT_COND_G - GOOD": 0,
    "INT_COND_A - AVERAGE": 0,
    "INT_COND_F - FAIR": 0,
    "INT_COND_P - POOR": 0,
    "OVERALL_COND_E - EXCELLENT": 1,
    "OVERALL_COND_VG - VERY GOOD": 0,
    "OVERALL_COND_G - GOOD": 0,
    "OVERALL_COND_A - AVERAGE": 0,
    "OVERALL_COND_F - FAIR": 0,
    "OVERALL_COND_P - POOR": 0,
    "KITCHEN_STYLE2_M - MODERN": 0,
    "KITCHEN_STYLE2_L - LUXURY": 1,
    "KITCHEN_STYLE2_S - SEMI-MODERN": 0,
    "KITCHEN_TYPE_F - FULL EAT IN": 1,
    "AC_TYPE_C - CENTRAL AC": 1,
    "AC_TYPE_D - DUCTLESS AC": 0,
    "HEAT_TYPE_F - FORCED HOT AIR": 0,
    "HEAT_TYPE_W - HT WATER/STEAM": 1,
    "PROP_VIEW_E - EXCELLENT": 1,
    "PROP_VIEW_G - GOOD": 0,
    "CORNER_UNIT_Y - YES": 1,
    "ORIENTATION_E - END": 1,
    "ORIENTATION_F - FRONT/STREET": 0,
    "EXT_COND_E - EXCELLENT": 1,
    "EXT_COND_G - GOOD": 0,
    "ROOF_COVER_S - SLATE": 1,
    "ROOF_COVER_A - ASPHALT SHINGL": 0
}
END_JSON

echo -e "\n${YELLOW}Testing prediction endpoint with alternative features...${NC}"
curl -s -X POST \
    -H "Content-Type: application/json" \
    -d @test_data2.json \
    http://localhost:$PORT/predict | python -m json.tool

# Create low-value property test data
echo -e "\n${YELLOW}Creating low-value property test data...${NC}"
cat > test_data3.json << 'END_JSON'
{
    "GROSS_AREA": 800.0,
    "LIVING_AREA": 600.0,
    "LAND_SF": 1200.0,
    "YR_BUILT": 1920,
    "YR_REMODEL": 1920,
    "BED_RMS": 1.0,
    "FULL_BTH": 1.0,
    "HLF_BTH": 0.0,
    "NUM_PARKING": 0,
    "FIREPLACES": 0,
    "KITCHENS": 1,
    "TT_RMS": 3,
    "ZIP_CODE": "02128",
    "STRUCTURE_CLASS_D - WOOD/FRAME": 1,
    "STRUCTURE_CLASS_C - BRICK/CONCR": 0,
    "STRUCTURE_CLASS_B - REINF CONCR": 0,
    "INT_COND_E - EXCELLENT": 0,
    "INT_COND_G - GOOD": 0,
    "INT_COND_A - AVERAGE": 0,
    "INT_COND_F - FAIR": 0,
    "INT_COND_P - POOR": 1,
    "OVERALL_COND_E - EXCELLENT": 0,
    "OVERALL_COND_VG - VERY GOOD": 0,
    "OVERALL_COND_G - GOOD": 0,
    "OVERALL_COND_A - AVERAGE": 0,
    "OVERALL_COND_F - FAIR": 0,
    "OVERALL_COND_P - POOR": 1,
    "KITCHEN_STYLE2_M - MODERN": 0,
    "KITCHEN_STYLE2_L - LUXURY": 0,
    "KITCHEN_STYLE2_S - SEMI-MODERN": 0,
    "KITCHEN_TYPE_F - FULL EAT IN": 0,
    "AC_TYPE_C - CENTRAL AC": 0,
    "AC_TYPE_D - DUCTLESS AC": 0,
    "HEAT_TYPE_F - FORCED HOT AIR": 0,
    "HEAT_TYPE_W - HT WATER/STEAM": 1,
    "PROP_VIEW_E - EXCELLENT": 0,
    "PROP_VIEW_G - GOOD": 0,
    "CORNER_UNIT_Y - YES": 0,
    "ORIENTATION_E - END": 0,
    "ORIENTATION_F - FRONT/STREET": 0,
    "EXT_COND_E - EXCELLENT": 0,
    "EXT_COND_G - GOOD": 0,
    "ROOF_COVER_S - SLATE": 0,
    "ROOF_COVER_A - ASPHALT SHINGL": 1
}
END_JSON

echo -e "\n${YELLOW}Testing prediction endpoint with low-value property...${NC}"
curl -s -X POST \
    -H "Content-Type: application/json" \
    -d @test_data3.json \
    http://localhost:$PORT/predict | python -m json.tool

# Cleanup temporary files
rm -f test_data.json test_data2.json test_data3.json

echo -e "\n${GREEN}Local testing complete!${NC}"
echo -e "You can now deploy to Cloud Run using: ./scripts/start_service.sh"
