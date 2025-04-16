#!/usr/bin/env python3
import json
import argparse
import requests
import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
if os.path.exists(".env"):
    load_dotenv()

def get_service_url(local=False):
    """Get the service URL"""
    if local:
        port = os.getenv("PORT", "8080")
        return f"http://localhost:{port}"
    
    # Get Cloud Run service URL
    region = os.getenv("REGION", "us-central1")
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "estateiqclone")
    cmd = f"gcloud run services describe estateiq-prediction --platform managed --region {region} --project {project_id} --format='get(status.url)'"
    import subprocess
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to get service URL: {result.stderr}")
    return result.stdout.strip()

def create_test_samples():
    """Create test samples with various feature combinations"""
    return [
        {
            # Typical single-family home
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
        },
        {
            # Luxury historic home
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
        },
        {
            # Low-value property
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
    ]

def test_prediction(url, features):
    """Test prediction endpoint"""
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(
            f"{url}/predict",
            json=features,
            headers=headers
        )
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
        
        return response.json()
    
    except Exception as e:
        print(f"Error making prediction request: {e}")
        return None

def test_health(url):
    """Test health endpoint"""
    try:
        response = requests.get(f"{url}/health")
        print("\nHealth Check:")
        print(f"Status Code: {response.status_code}")
        if response.ok:
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error checking health: {e}")

def test_metadata(url):
    """Test metadata endpoint"""
    try:
        response = requests.get(f"{url}/metadata")
        print("\nMetadata:")
        print(f"Status Code: {response.status_code}")
        if response.ok:
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error fetching metadata: {e}")

def main():
    parser = argparse.ArgumentParser(description='Test EstateIQ prediction service')
    parser.add_argument('--url', help='Service URL (optional)')
    parser.add_argument('--local', action='store_true', help='Use local service')
    args = parser.parse_args()

    # Get service URL
    service_url = args.url or get_service_url(args.local)
    print(f"Using service URL: {service_url}")

    # Test health endpoint
    test_health(service_url)

    # Test metadata endpoint
    test_metadata(service_url)

    # Create test samples
    test_samples = create_test_samples()
    print(f"\nCreated {len(test_samples)} test samples")

    # Test predictions
    print("\nTesting predictions:")
    test_descriptions = [
        "Typical single-family home (good condition)",
        "Luxury historic home (excellent condition)",
        "Low-value property (poor condition)"
    ]
    
    for idx, (description, features) in enumerate(zip(test_descriptions, test_samples)):
        print(f"\n{'-'*80}")
        print(f"Test Case {idx + 1}: {description}")
        print(f"{'-'*80}")
        print("\nKey Features:")
        print(f"  Area: {features['GROSS_AREA']} sq ft ({features['LIVING_AREA']} sq ft living)")
        print(f"  Rooms: {features['TT_RMS']} total, {features['BED_RMS']} bedrooms")
        print(f"  Built: {features['YR_BUILT']}, Last Remodel: {features['YR_REMODEL']}")
        print(f"  Location: ZIP {features['ZIP_CODE']}")
        
        prediction = test_prediction(service_url, features)
        if prediction:
            print("\nPrediction Results:")
            print(f"  Estimated Value: ${prediction['prediction']:,.2f}")
            print(f"  Timestamp: {prediction['timestamp']}")
            print(f"  Model Version: {prediction['model_version']}")

if __name__ == "__main__":
    main()
