import os
import json
import joblib
import logging
import subprocess
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_model_artifacts():
    """Download model and metrics from GCP."""
    bucket = os.getenv('MODEL_ARTIFACTS_BUCKET')
    model_path = os.getenv('MODEL_REGISTRY_PATH')
    model_dir = Path(os.getenv('MODEL_DIR', '/app/models'))
    
    if not all([bucket, model_path]):
        raise ValueError("Required environment variables not set")
    
    current_model_path = f"gs://{bucket}/{model_path}/current"
    model_file = model_dir / "current" / "model.joblib"
    metrics_file = model_dir / "current" / "metrics.json"
    
    # Ensure directories exist
    model_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download model file
        logger.info(f"Downloading model from {current_model_path}")
        subprocess.run([
            "gsutil",
            "cp",
            f"{current_model_path}/model.joblib",
            str(model_file)
        ], check=True)
        
        # Download metrics file
        subprocess.run([
            "gsutil",
            "cp",
            f"{current_model_path}/metrics.json",
            str(metrics_file)
        ], check=True)
        
        # Verify model file exists and is valid
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found at {model_file}")
        
        # Load and verify model
        try:
            model = joblib.load(model_file)
            logger.info("Model loaded successfully")
            
            # Log model metrics
            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics = json.load(f)
                logger.info(f"Model metrics: {metrics}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download artifacts: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

def main():
    """Main entry point for model loading."""
    try:
        download_model_artifacts()
        logger.info("Model artifacts downloaded and verified successfully")
    except Exception as e:
        logger.error(f"Failed to download and verify model artifacts: {e}")
        raise

if __name__ == "__main__":
    main()
