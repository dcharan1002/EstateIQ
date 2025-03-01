import os
import requests
from pathlib import Path

def download_boston_housing_data(output_dir: str = "/opt/airflow/data/raw") -> str:
    """Downloads Boston housing dataset"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # URL for Boston Housing dataset
    url = "https://placekey-free-datasets.s3.us-west-2.amazonaws.com/boston-property-assessment-data/csv/boston-property-assessment-data.csv"
    output_path = os.path.join(output_dir, "boston_2025.csv")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Save the downloaded data using streaming to handle large files
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
            
        print(f"Data downloaded successfully to {output_path}")
        return output_path
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading data: {e}")
        raise
