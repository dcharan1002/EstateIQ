import os
import joblib
import logging
import json
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from pathlib import Path
from datetime import datetime
from flask_cors import CORS


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Model loading
MODEL_DIR = Path(os.getenv('MODEL_DIR', '/app/models'))
MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "artifacts" / "model_latest.joblib"
METRICS_PATH = MODEL_DIR / "current" / "metrics.json"

def load_model():
    """Load the model and return it."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    
    try:
        model = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

try:
    model = load_model()
    # Get expected features
    if hasattr(model, 'feature_names_in_'):
        EXPECTED_FEATURES = model.feature_names_in_
    elif hasattr(model, 'get_booster'):
        EXPECTED_FEATURES = model.get_booster().feature_names
    else:
        EXPECTED_FEATURES = []
except Exception as e:
    logger.error(f"Failed to load model on startup: {e}")
    raise

def get_feature_defaults():
    """Get default values for each feature type"""
    return {
        'GROSS_AREA': 2000.0,
        'LIVING_AREA': 1500.0,
        'LAND_SF': 5000.0,
        'YR_BUILT': 1980,
        'YR_REMODEL': 1980,
        'BED_RMS': 3.0,
        'FULL_BTH': 2.0,
        'HLF_BTH': 1.0,
    }

def prepare_features(input_data):
    """Prepare input features with defaults for missing values"""
    # Get default values
    defaults = {feature: 0.0 for feature in EXPECTED_FEATURES}
    feature_defaults = get_feature_defaults()
    defaults.update(feature_defaults)

    # Create feature vector with defaults
    features = {}
    for feature in EXPECTED_FEATURES:
        if feature in input_data:
            features[feature] = input_data[feature]
        else:
            features[feature] = defaults[feature]
            
    return pd.DataFrame([features])
@app.route('/')
def index():
    return "Welcome to EstateIQ API!"


@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions using the loaded model."""
    try:
        # Get data from request
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Create DataFrame with defaults for missing features
        try:
            features = prepare_features(data)
            # Replace NaN and inf values with 0
            features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
        except Exception as e:
            return jsonify({"error": f"Invalid data format: {str(e)}"}), 400

        # Make prediction
        prediction = model.predict(features)

        # Log prediction details
        logger.info(f"Made prediction: {prediction[0]} for features: {data}")

        # Return prediction with timestamp and model info
        response = {
            "prediction": float(prediction[0]),
            "timestamp": datetime.now().isoformat(),
            "model_version": os.getenv('BUILD_ID', 'unknown')
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint that also verifies model is loaded."""
    try:
        # Verify model is loaded
        if model is None:
            return jsonify({
                "status": "unhealthy",
                "error": "Model not loaded"
            }), 503

        # Return healthy status with model info
        return jsonify({
            "status": "healthy",
            "model": {
                "path": str(MODEL_PATH),
                "version": os.getenv('BUILD_ID', 'unknown'),
                "loaded": True
            }
        })

    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 503

@app.route('/metadata', methods=['GET'])
def metadata():
    """Return model metadata and performance metrics."""
    try:
        if METRICS_PATH.exists():
            with open(METRICS_PATH) as f:
                metrics = json.load(f)
        else:
            metrics = {"warning": "No metrics file found"}

        return jsonify({
            "model_version": os.getenv('BUILD_ID', 'unknown'),
            "metrics": metrics,
            "deployment_timestamp": os.getenv('DEPLOYMENT_TIMESTAMP', 'unknown')
        })

    except Exception as e:
        logger.error(f"Error fetching metadata: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 3000))
    app.run(host='0.0.0.0', port=port)
