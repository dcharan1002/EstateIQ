#!/usr/bin/env python3
import os
import json
import numpy as np
import argparse
import requests
import pandas as pd
from pathlib import Path

def get_service_url(local=False):
    """Get the service URL"""
    if local:
        return "http://localhost:8080"
    
    # Get Cloud Run service URL
    cmd = "gcloud run services describe estateiq-prediction --platform managed --region us-central1 --format='get(status.url)'"
    import subprocess
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to get service URL: {result.stderr}")
    return result.stdout.strip()

def create_test_samples():
    """Create test samples with various feature combinations"""
    return [
        {
            # Basic features
            "GROSS_AREA": 2500,
            "LIVING_AREA": 2000,
            "LAND_SF": 5000,
            "YR_BUILT": 1990,
            "BED_RMS": 3,
            "FULL_BTH": 2,
            "HLF_BTH": 1
        },
        {
            # Minimal features
            "GROSS_AREA": 3000,
            "LIVING_AREA": 2500,
            "LAND_SF": 6000
        },
        {
            # Features with condition
            "GROSS_AREA": 2800,
            "LIVING_AREA": 2200,
            "STRUCTURE_CLASS_C - BRICK/CONCR": 1,
            "INT_COND_A - AVERAGE": 1,
            "OVERALL_COND_A - AVERAGE": 1
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
    for idx, features in enumerate(test_samples):
        print(f"\nSample {idx + 1}:")
        print("Features:", json.dumps(features, indent=2))
        prediction = test_prediction(service_url, features)
        if prediction:
            print("Prediction:", json.dumps(prediction, indent=2))

if __name__ == "__main__":
    main()
